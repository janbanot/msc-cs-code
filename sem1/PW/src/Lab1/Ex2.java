package Lab1;

class Singer extends Thread {

    private final int n;

    public Singer(int n) {
        this.n = n;
    }

    @Override
    public void run() {
        System.out.println(this.n + " bottles of beer on the wall, " + this.n + " bottles of beer");
        System.out.println("Take one down and pass it around, " + (this.n + 1) + " bottles of beer on the wall");
    }
}

public class Ex2 {
    public static void main(String[] args) {
        Singer[] threads = new Singer[100];
        for (int i = 0; i < threads.length; ++i) {
            threads[i] = new Singer(i + 1);
        }
        for (Thread t : threads) {
            t.start();
        }
    }
}