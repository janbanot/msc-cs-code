package Lab3;

import java.util.concurrent.Semaphore;

class FooBar {
    private final int n;

    private final Semaphore foo = new Semaphore(1);
    private final Semaphore bar = new Semaphore(0);

    public FooBar(int n) {
        this.n = n;
    }

    public void foo() throws InterruptedException {
        for (int i = 0; i < n; i++) {
            foo.acquire();
            System.out.print("foo");
            bar.release();
        }
    }

    public void bar() throws InterruptedException {
        for (int i = 0; i < n; i++) {
            bar.acquire();
            System.out.println("bar");
            foo.release();
        }
    }
}

public class Ex2 {
    public static void main(String[] args) {
        // Drukowanych będzie 40 komunikatów
        final FooBar foobar = new FooBar(40);
        // Pierwszy wątek -- wywołuje foo()
        new Thread(new Runnable() {

            public void run() {
                try {
                    foobar.foo();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
        // Drugi wątek -- wywołuje bar()
        new Thread(new Runnable() {
            public void run() {
                try {
                    foobar.bar();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }).start();
    }
}
