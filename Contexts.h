#ifndef EX3_CONTEXTS_H
#define EX3_CONTEXTS_H


#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include <atomic>
#include <pthread.h>
#include <vector>
#include <semaphore.h>
#include "Barrier.h"




struct JobContext;

struct ThreadContext;


/**
 * A struct for saving relevant information for (context) for
 * each thread (to pass as arg to its main function).
 */
struct ThreadContext {

    JobContext &job_context;

    IntermediateVec intermediate_vec;

    /**
     * Construct a ThreadContext with the given parent job.
     * @param job_context
     */
    explicit ThreadContext(JobContext &job_context) : job_context(job_context) {}
};


/**
 * An object for storing all relevant job information.
 */
struct JobContext{

    /** The input data for the job */
    const InputVec& input_vec;
    /** MapReduceClient with the map and reduce functions */
    const MapReduceClient& client;
    /** Vector to store the output. */
    OutputVec& output_vec;
    /** Vector for each key from map phase saved for reduce phase. */
    std::vector<IntermediateVec> to_reduce_queue;



    /** The collection of threads. */
    std::vector<pthread_t> threads;
    /** Context objects for each thread. */
    std::vector<ThreadContext> thread_contexts;



    /** Multivar atomic variable for job state. */
    std::atomic<uint64_t> job_state;
    /** Number of elements already mapped */
    std::atomic<std::size_t> num_of_mapped_elements;
    /** Total number of elements for reduce phase */
    std::atomic<std::size_t> num_of_intermediate_elements;
    /**An atomic counter for the number of vectors reduces so far. */
    std::atomic<std::size_t> num_of_reduced_vectors;


    /** Barrier for stopping before shuffle. */
    Barrier barrier_before_shuffle;
    /** Semaphore for waiting until shuffle ends */
    sem_t wait_for_shuffle_semaphore{};
    /** Mutex for protecting writing into output vector. */
    pthread_mutex_t protect_output_mutex;
    /** Mutex for managing waiting for job */
    pthread_mutex_t wait_for_job_mutex;
    /** An indicator for managing waiting for job */
    bool threads_waiting = false;

    /** Create and initialize job context with the given information */
    JobContext(const InputVec& inputVec, const MapReduceClient& client,
               OutputVec& output_vec, std::size_t num_of_threads);

    /** Make JobContext Non-copyable */
    JobContext(const JobContext& ) = delete; // non construction-copyable
    JobContext& operator=( const JobContext& ) = delete;  // non copyable
    JobContext(const JobContext&& ) = delete;  // non construction-movable.
    JobContext& operator=(const JobContext&&) = delete; // Non-movable.

    /** Destroy semaphores and mutexes */
    ~JobContext();
};


#endif //EX3_CONTEXTS_H
