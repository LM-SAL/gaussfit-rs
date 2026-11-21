use crate::workers::{run, WorkerPool};
use rayon::ThreadPoolBuilder;

#[test]
fn the_kernel_inherits_the_selected_pool() {
    let pool = WorkerPool(ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    assert_eq!(run(Some(&pool), rayon::current_num_threads), 2);
    pool.0
        .install(|| assert_eq!(run(None, rayon::current_num_threads), 2));
}
