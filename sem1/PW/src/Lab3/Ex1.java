package Lab3;

import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

class Parking {
    private int capacity;
    private AtomicInteger carsParked = new AtomicInteger(0);
    // TODO - Kod rozwiązania...
    private Semaphore capacitySemaphore = null;

    public Parking(int capacity) {
        this.capacity = capacity;
        // TODO - Kod rozwiązania...
        capacitySemaphore = new Semaphore(capacity - 1);
    }

    public void enter() throws InterruptedException {
        // TODO - Kod rozwiązania...
        capacitySemaphore.acquire();
        if (carsParked.incrementAndGet() > capacity) {
            throw new RuntimeException("Too many cars!");
        }
    }

    public void leave() {
        carsParked.decrementAndGet();
        // TODO - Kod rozwiązania...
        capacitySemaphore.release();

    }
}

class Car extends Thread {
    private Parking parking;
    private int id;
    private int attempts;

    public Car(Parking parking, int id, int attempts) {
        this.parking = parking;
        this.id = id;
        this.attempts = attempts;
    }

    private void rest() throws InterruptedException {
        Thread.sleep(1000 * (1 + ThreadLocalRandom.current().nextInt(2)));
    }

    @Override
    public void run() {
        try {
            for (int i = 0; i < attempts; ++i) {
                rest();
                System.out.printf("[Car %-3d] arrives\n", id);
                parking.enter();
                rest();
                parking.leave();
                System.out.printf("[Car %-3d] leaves (parked: %d times)\n", id, i + 1);
            }
        } catch (InterruptedException e) {
        }
    }
}

public class Ex1 {
    public static void main(String[] args) throws InterruptedException {
        Parking parking = new Parking(4);
        int parkAttempts = 5;
        Thread[] cars = new Thread[6];
        for (int i = 0; i < cars.length; ++i) {
            cars[i] = new Car(parking, i + 1, parkAttempts);
        }
        System.out.println("Simulation started.");
        for (Thread t : cars) {
            t.start();
        }
        for (Thread t : cars) {
            t.join();
        }
        System.out.println("Simulation finished.");
    }
}

