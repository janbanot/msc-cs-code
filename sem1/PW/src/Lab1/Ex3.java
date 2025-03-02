package Lab1;

class ExtendedSinger extends Thread {

    private final int n;
    private ExtendedSinger next;

    public ExtendedSinger(int n) {
        this.n = n;
    }

    public void setNext(ExtendedSinger next) {
        this.next = next;
    }

    @Override
    public void run() {
        try {
            if (next != null) {
                next.join();
            }
            System.out.println(this.n + " bottles of beer on the wall, " + this.n + " bottles of beer");
            System.out.println("Take one down and pass it around, " + (this.n - 1) + " bottles of beer on the wall");
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}

public class Ex3 {
    public static void main(String [] args) {
        ExtendedSinger [] threads = new ExtendedSinger[101];
        for (int i = 100; i >= 0; --i) {
            threads[i] = new ExtendedSinger(i);
            if (i < 100) {
                threads[i].setNext(threads[i + 1]);
            }
        }
        for (Thread t : threads) {
            t.start();
        }
    }
}