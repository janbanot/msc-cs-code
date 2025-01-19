package Lab5;

import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;
import java.util.ArrayList;
import java.util.List;

class Task extends RecursiveTask<Long> {
    final long n, m;
    private static final long THRESHOLD = 1000;
    
    public Task(long n, long m) { 
        this.n = n; 
        this.m = m; 
    }

    public long countPartitionsSeq(long n, long m) {
        if (n <= 1 || m == 1) return 1;
        long count = 0;
        for (long k = Math.min(n, m); k >= 1; --k) {
            count += countPartitionsSeq(n-k, Math.min(k, m));
        }
        return count;
    }

    @Override
    public Long compute() {
        if (n <= 1 || m == 1) return 1L;

        if (n * m < THRESHOLD) {
            return countPartitionsSeq(n, m);
        }
        
        long count = 0;
        long min = Math.min(n, m);

        List<Task> tasks = new ArrayList<>();
        for (long k = min; k >= 1; k--) {
            Task task = new Task(n - k, Math.min(k, m));
            task.fork();
            tasks.add(task);
        }

        for (Task task : tasks) {
            count += task.join();
        }
        
        return count;
    }
}

public class Ex3 {
    public static void main(String [] args) {
        long n = 145; long m = n;
        // Sekwencyjnie
        long start_time = System.nanoTime();
        long ans = new Task(n, m).countPartitionsSeq(n, m);
        long end_time = System.nanoTime();
        System.out.printf("The answer is: %d\nComputation time: %.1f sec.\n",
                ans, (end_time - start_time) * 1e-9);
        // Równolegle
        ForkJoinPool pool = new ForkJoinPool();
        start_time = System.nanoTime();
        ans = pool.invoke(new Task(n, m));
        end_time = System.nanoTime();
        System.out.printf("The answer is: %d\nComputation time: %.1f sec.\n",
                ans, (end_time - start_time) * 1e-9);
    }
}
