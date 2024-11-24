package Lab3;

import java.util.concurrent.Semaphore;

class Pot {
    static final int M = 5; // Pojemność kotła
    private Semaphore emptyPot = new Semaphore(1);
    private Semaphore available = new Semaphore(0);
    private int servingsAvailable = 0;
    private int totalServedCount = 0;

    private synchronized void insertServings(int value) {
        servingsAvailable = value;
    }

    private synchronized int removeServing() {
        --servingsAvailable;
        ++totalServedCount;
        return servingsAvailable;
    }

    public int getTotalServedCount() {
        return totalServedCount;
    }

    public void getServing() throws InterruptedException {
        // Wait for available portions
        available.acquire();

        // Remove one serving and get remaining count
        int remaining = removeServing();

        // If this was the last portion, wake up the cook
        if (remaining == 0) {
            emptyPot.release();  // Signal cook to fill pot
        }
    }

    public void fill() throws InterruptedException {
        // Wait for empty pot signal
        emptyPot.acquire();

        // Fill the pot with M portions
        insertServings(M);

        // Release M permits to allow M servings to be taken
        for (int i = 0; i < M; i++) {
            available.release();
        }

    }
}

class Person extends Thread { // Reprezentuje tubylca
    Pot pot;
    int servingsConsumed = 0;

    public Person(String name, Pot pot) {
        super(name);
        this.pot = pot;
    }

    @Override
    public void run() {
        try {
            for (int i = 0; i < 100; ++i) {
                pot.getServing();
                ++servingsConsumed;
                Thread.yield();
            }
        } catch (InterruptedException e) {
            return;
        }
    }
}

class Cook extends Thread { // Reprezentuje kucharza
    Pot pot;

    public Cook(Pot pot) {
        this.pot = pot;
        setDaemon(true);
    }

    @Override
    public void run() {
        try {
            while (!isInterrupted()) {
                pot.fill();
            }
        } catch (InterruptedException e) {
            return;
        }
    }
}

public class Ex4 {
    public static void main(String[] args) throws InterruptedException {
        Pot pot = new Pot();
        Cook cook = new Cook(pot);
        final int N = 10;
        Person[] people = new Person[N];
        for (int i = 0; i < people.length; ++i) {
            people[i] = new Person("Person " + i, pot);
        }
        cook.start();
        for (Thread t : people) {
            t.start();
        }
        for (Thread t : people) {
            t.join();
        }
        cook.interrupt();
        System.out.printf("Total served: %d.\n", pot.getTotalServedCount());
        for (Person p : people) {
            System.out.printf("[%s] Ate %d servings.\n", p.getName(), p.servingsConsumed);
        }
        System.out.println("Finishing simulation.");
    }
}
