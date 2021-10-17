#include "MapReduceFramework.h"
#include "Barrier.h"
#include <atomic>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "Contexts.h"

/** Error messages */
#define PTHREAD_ERR "Pthread failed. "
#define MEM_ERR "system error: Memory allocation failed."

/** Wrap c-functions error checking and exit to avoid code clutter. */
#define CHECKED_CALL(call, msg) if (call != 0){ \
                                    std::cerr << "system error: " << msg << std::endl; \
                                    std::exit(EXIT_FAILURE);\
}


/**
 * Macros for managing multiple fields of job state saved as bits
 * of atomic 64 bit integer.
 */
#define GET_TOTAL_TO_PROCESS(job_state) ((job_state << 2U) >> 33U)
#define GET_ALREADY_PROCESSED(job_state) (job_state & NUM_OF_DONE_BITS)
#define SET_INITIALIZE_JOB_STATE(job_state, n) (job_state |= (uint64_t(n) << 31U))
#define GET_JOB_STAGE(job_state) (job_state >> 62U)
static const uint64_t ONE_IN_STAGE_BITS = uint64_t(1) << 62U;
static const uint64_t NUM_OF_DONE_BITS = (uint64_t(1) << 31U) - 1;


/** Run map and reduce and maybe shuffle in the middle (depends on the thread) */
template <bool should_shuffle>
void* map_reduce(void* arg);

/** Map client.map function to input vector. */
void map(ThreadContext& thread_context);

/** Shuffle intermediate vectors produced in map phase. */
void shuffle(ThreadContext& context);

/** Call client.reduce function on the vectors by key. */
void reduce(ThreadContext& context);

// ----------- Implementation of JobContext function (see docs in Contexts.h header). ---------

JobContext::JobContext(const InputVec& input_vec, const MapReduceClient& client,
                       OutputVec& output_vec, std::size_t num_of_threads):
        input_vec(input_vec), client(client), output_vec(output_vec),
        threads(num_of_threads), job_state(0), num_of_mapped_elements(0),
        num_of_intermediate_elements(0), num_of_reduced_vectors(0),
        barrier_before_shuffle(num_of_threads),
        protect_output_mutex(PTHREAD_MUTEX_INITIALIZER),
        wait_for_job_mutex(PTHREAD_MUTEX_INITIALIZER)


{
    SET_INITIALIZE_JOB_STATE(job_state, input_vec.size());
    CHECKED_CALL(sem_init(&wait_for_shuffle_semaphore, 0, 0), PTHREAD_ERR)
    thread_contexts.reserve(num_of_threads);
    threads.reserve(num_of_threads);
    thread_contexts.emplace_back(*this);
    CHECKED_CALL(pthread_create(&(threads[0]), nullptr, map_reduce<true>,
                                    static_cast<void *>(&(thread_contexts[0]))), PTHREAD_ERR)
    for (std::size_t i = 1; i < num_of_threads; i++){
        thread_contexts.emplace_back(*this);
        CHECKED_CALL(pthread_create(&(threads[i]), nullptr, map_reduce<false>,
                                            static_cast<void *>(&(thread_contexts[i]))), PTHREAD_ERR)
    }
};


JobContext::~JobContext() {
        CHECKED_CALL(sem_destroy(&wait_for_shuffle_semaphore), PTHREAD_ERR)
        CHECKED_CALL(pthread_mutex_destroy(&protect_output_mutex), PTHREAD_ERR)
        CHECKED_CALL(pthread_mutex_destroy(&wait_for_job_mutex), PTHREAD_ERR)
}

/**
 * Start running MapReduce job and return job handle.
 * @param client An implementation of MapReduceClient.h (map and reduce functions)
 * @param inputVec Input elements to process.
 * @param outputVec Vector to store the results.
 * @param multiThreadLevel Number of threads to use for job.
 * @return Job handle.
 */
JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
    try {
        return static_cast<JobHandle>(new JobContext(inputVec, client, outputVec, multiThreadLevel));
    } catch (const std::bad_alloc& e) {
        std::cerr << MEM_ERR << std::endl;
        std::exit(EXIT_FAILURE);
    }

}


/**
 * Add (key,value) into an intermediate vector based on context.
 * @param key
 * @param value
 * @param context
 */
void emit2 (K2* key, V2* value, void* context){
    ThreadContext& thread_context (*static_cast<ThreadContext*>(context));
    thread_context.intermediate_vec.push_back({key, value});
    thread_context.job_context.num_of_intermediate_elements++;
}

/**
 * Add (key, val) to output vector.
 * @param key
 * @param value
 * @param context
 */
void emit3 (K3* key, V3* value, void* context){
    ThreadContext& thread_context(*static_cast<ThreadContext*>(context));
    CHECKED_CALL(pthread_mutex_lock(&thread_context.job_context.protect_output_mutex), PTHREAD_ERR)
    thread_context.job_context.output_vec.push_back({key, value});
    CHECKED_CALL(pthread_mutex_unlock(&thread_context.job_context.protect_output_mutex), PTHREAD_ERR)
}


/** Get the stage and percentage of the job and store it in the state struct. */
void getJobState(JobHandle job, JobState* state){
    uint64_t job_state = static_cast<JobContext*>(job)->job_state;
    state->stage = static_cast<stage_t>(GET_JOB_STAGE(job_state));
    state->percentage = 100.f *(GET_ALREADY_PROCESSED(job_state)
            / (float)std::max(GET_TOTAL_TO_PROCESS(job_state), uint64_t(1)));
}


/**
 * Block the calling thread until the job is done.
 * @param job The handle for the job.
 */
void waitForJob(JobHandle job){
    JobContext& job_context = *static_cast<JobContext*>(job);
    CHECKED_CALL(pthread_mutex_lock(&job_context.wait_for_job_mutex), PTHREAD_ERR)
    if (!job_context.threads_waiting){
        job_context.threads_waiting = true;
        for (const auto& thread:job_context.threads){
            CHECKED_CALL(pthread_join(thread, nullptr), PTHREAD_ERR)
        }
    }
    CHECKED_CALL(pthread_mutex_unlock(&job_context.wait_for_job_mutex), PTHREAD_ERR)
}


/** Wait for job to finish and release all resources. */
void closeJobHandle(JobHandle job){
    waitForJob(job);
    delete static_cast<JobContext*>(job);
}


// -------------- Implementation of helper function declared above (doc is there) ----------

template <bool should_shuffle>
void* map_reduce(void* arg){
        ThreadContext& context(*static_cast<ThreadContext*>(arg));
        context.job_context.job_state |= ONE_IN_STAGE_BITS;
        map(context);
        std::sort(context.intermediate_vec.begin(), context.intermediate_vec.end(),
                  [&] (const IntermediatePair& a, const IntermediatePair& b) {return *a.first < *b.first; });
        context.job_context.barrier_before_shuffle.barrier();
        if (should_shuffle){  // Main thread
            shuffle(context);
        } else{ // Other thread
            CHECKED_CALL(sem_wait(&context.job_context.wait_for_shuffle_semaphore), PTHREAD_ERR)
        }
        CHECKED_CALL(sem_post(&context.job_context.wait_for_shuffle_semaphore), PTHREAD_ERR)
        reduce(context);
    return nullptr;
}


void map(ThreadContext& thread_context){
    std::atomic<std::size_t>& processed(thread_context.job_context.num_of_mapped_elements);
    std::size_t input_size(thread_context.job_context.input_vec.size());
    for (std::size_t i = processed++; i < input_size; i = processed++) {
        auto &pair = thread_context.job_context.input_vec[i];
        thread_context.job_context.client.map(pair.first, pair.second, static_cast<void *>(&thread_context));
        thread_context.job_context.job_state++;
    }
}


void reduce(ThreadContext& context){
    std::atomic<std::size_t>& processed(context.job_context.num_of_reduced_vectors);
    std::size_t queue_size(context.job_context.to_reduce_queue.size());
    for (std::size_t i = processed++; i < queue_size; i = processed++) {
        auto &to_reduce = context.job_context.to_reduce_queue[i];
        context.job_context.client.reduce(&to_reduce, static_cast<void *>(&context));
        context.job_context.job_state += to_reduce.size();
    }
}



void shuffle(ThreadContext& context){
    // We shuffle the vectors by first finding the maximal key
    // out of the keys in the end of the sorted vectors.
    // for each vector we pop elements while their key is the maximal
    // key we found and reduce. Now that all elements with this key
    // Are processed we star over and search for the new maximum.
    context.job_context.job_state =
            (uint64_t(context.job_context.num_of_intermediate_elements) << 31U) + (ONE_IN_STAGE_BITS << 1U);
    std::vector<ThreadContext>& contexts(context.job_context.thread_contexts);
    std::vector<IntermediateVec*> intermediate_vectors(contexts.size());
    std::transform(contexts.begin(), contexts.end(), intermediate_vectors.begin(),
                   [](ThreadContext& thread_context) { return &thread_context.intermediate_vec;});
    for(auto num_left = static_cast<std::size_t>(context.job_context.num_of_intermediate_elements); num_left > 0;){
        auto compare_last_element = [](const IntermediateVec* x, const IntermediateVec* y) {
            return (x->empty()) || (!y->empty() && (*((x->end() - 1)->first) < *((y->end() - 1)->first)));
        };
        auto it_to_max_vector = std::max_element(intermediate_vectors.begin(), intermediate_vectors.end(),
                                                 compare_last_element);
        K2& maximal_key = *(((*it_to_max_vector)->end() - 1)->first);
        IntermediateVec max_key_vec;
        for (auto p_vec:intermediate_vectors){
            while (!p_vec->empty() && !(*((p_vec->end() - 1)->first) < maximal_key)){
                max_key_vec.push_back(*(p_vec->end() - 1));
                p_vec->pop_back();
                num_left--;
                context.job_context.job_state++;
            }
        }
        context.job_context.to_reduce_queue.push_back(max_key_vec);
    }
    context.job_context.job_state = (uint64_t(context.job_context.num_of_intermediate_elements) << 31U)
            + (ONE_IN_STAGE_BITS << 1U) + ONE_IN_STAGE_BITS;
}




