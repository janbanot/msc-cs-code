package Lab3;

import java.util.concurrent.Semaphore;

class SystemServices {
    private int n;
    // Semafory do użycia w metodach log, processData, networkTransmit
    Semaphore logSemaphore = new Semaphore(1);
    Semaphore processSemaphore = new Semaphore(0);
    Semaphore networkSemaphore = new Semaphore(0);

    public SystemServices(int n) {
        this.n = n;
    }

    public void log() throws InterruptedException {
        for (int i = 0; i < n; i++) {
            logSemaphore.acquire();
            System.out.println("Logowanie: Wpis danych " + i);
            processSemaphore.release();
        }
    }

    public void processData() throws InterruptedException {
        for (int i = 0; i < n; i++) {
            processSemaphore.acquire();
            System.out.println("Przetwarzanie: Przetwarzanie danych " + i);
            networkSemaphore.release();
        }
    }

    public void networkTransmit() throws InterruptedException {
        for (int i = 0; i < n; i++) {
            networkSemaphore.acquire();
            System.out.println("Sieć: Transmisja danych " + i);
            logSemaphore.release();
        }
    }
}

public class Ex3 {
    public static void main(String[] args) {
// Należy przetworzyć 10 sekwencji
        final SystemServices services = new SystemServices(10);
// Wątek logowania
        new Thread(() -> {
            try {
                services.log();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }).start();
// Wątek przetwarzania danych
        new Thread(() -> {
            try {
                services.processData();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }).start();
// Wątek transmisji sieciowej
        new Thread(() -> {
            try {
                services.networkTransmit();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }).start();
    }
}
