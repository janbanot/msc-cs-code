package Lab1;

class Bob extends Thread {
    private int input;

    public void setInput(int input) {
        this.input = input;
        this.interrupt();
    }

    @Override
    public void run() {
        while (true) {
            if (Thread.interrupted()) {
                if (input == 0) {
                    System.out.println("[Bob] Finishing work.");
                    break;
                } else {
                    System.out.println("[Bob] The result is: " + (input * 2));
                }
            }
        }
    }
}

class Alice extends Thread {
    private final Bob bob;

    public Alice(Bob bob) {
        this.bob = bob;
    }

    @Override
    public void run() {
        try {
            for (int i = 1; i <= 10; ++i) {
                System.out.println("[Alice] Sending to Bob: " + i);
                bob.setInput(i);
                Thread.sleep(1000);
            }
            bob.setInput(0);

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}


public class Ex5 {
    public static void main(String[] args) {
        Bob bob = new Bob();
        Alice alice = new Alice(bob);
        bob.start();
        alice.start();
    }
}
