package Lab1;

class Worker extends Thread {

    private final Number sleepTime;

    public Worker(String name, Number sleepTime) {
        super(name);
        this.sleepTime = sleepTime;
    }

    @Override
    public void run() {
        try {
            System.out.println("Hello from: " + this.getName());
            Thread.sleep(this.sleepTime.longValue());
            System.out.println("Bye from: " + this.getName());
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}

public class Ex1 {
    public static void main(String [] args) {
        Thread alice = new Worker("Alice", 1000);
        Thread bob = new Worker("Bob", 1000);
        Thread charles = new Worker("Charles", 1000);

        alice.start();
        bob.start();
        charles.start();
    }
}
