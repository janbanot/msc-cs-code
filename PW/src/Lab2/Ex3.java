package Lab2;

import java.util.concurrent.locks.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.Random;

class Bank {
    public static final int N = 10;
    private int [] balances = new int[N];
    private Lock [] locks = new Lock[N];

    public Bank() {
        for (int i = 0; i < locks.length; ++i) {
            locks[i] = new ReentrantLock();
        }
    }

    public void deposit(int accountId, int amount) {
        balances[accountId] += amount;
    }
    public int getBalance(int accountId) {
        return balances[accountId];
    }

    public boolean transfer(int fromAccount, int toAccount, int amount) {
        // Always acquire locks in a consistent order to prevent deadlocks
        int firstLock = Math.min(fromAccount, toAccount);
        int secondLock = Math.max(fromAccount, toAccount);

        // Acquire first lock
        locks[firstLock].lock();
        try {
            // Acquire second lock
            locks[secondLock].lock();
            try {
                // Check if there's enough money in the source account
                if (balances[fromAccount] >= amount) {
                    balances[fromAccount] -= amount;
                    balances[toAccount] += amount;
                    return true;
                }
                return false;
            } finally {
                locks[secondLock].unlock();
            }
        } finally {
            locks[firstLock].unlock();
        }
    }

    public void equalize(int accountA, int accountB) {
        // Always acquire locks in a consistent order to prevent deadlocks
        int firstLock = Math.min(accountA, accountB);
        int secondLock = Math.max(accountA, accountB);

        // Try to acquire both locks without blocking
        if (locks[firstLock].tryLock()) {
            try {
                if (locks[secondLock].tryLock()) {
                    try {
                        int total = balances[accountA] + balances[accountB];
                        balances[accountA] = total / 2 + total % 2;
                        balances[accountB] = total / 2;
                    } finally {
                        locks[secondLock].unlock();
                    }
                }
            } finally {
                locks[firstLock].unlock();
            }
        }
    }
}

class Accountant extends Thread {
    Bank bank;
    public Accountant(Bank bank) { this.bank = bank; }
    @Override
    public void run() {
        Random rng = ThreadLocalRandom.current();
        for (int i = 0; i < 1000; ++i) {
            // Try to transfer a random amount between a pair of accounts
            // The accounts numbers (ids) are also selected randomly
            int fromAccount = rng.nextInt(Bank.N-1);
            int toAccount = rng.nextInt(Bank.N-1);
            while (toAccount == fromAccount) { // Source should differ from
                // the target
                toAccount = rng.nextInt(Bank.N-1); // Try again
            }
            if (rng.nextBoolean()) { // 50% of the time we transfer...
                bank.transfer(fromAccount, toAccount, rng.nextInt(100));
            } else { // ...the remaining 50% of the time we equalize
                bank.equalize(fromAccount, toAccount);
            }
        }
    }
}

public class Ex3 {
    public static void main(String [] args) throws InterruptedException {
        Bank bank = new Bank();
        for (int i = 0; i < Bank.N; ++i) {
            bank.deposit(i, 100);
        }
        Thread [] threads = new Thread[10];
        for (int i = 0; i < threads.length; ++i) {
            threads[i] = new Accountant(bank);
        }
        for (Thread t : threads) { t.start(); }
        for (Thread t : threads) { t.join(); }
        int total = 0;
        for (int i = 0; i < Bank.N; ++i) {
            int b = bank.getBalance(i);
            total += b;
            System.out.printf("Account [%d] balance: %d\n", i, b);
        }
        System.out.printf("Total balance is %d\tvalid value is %d\n",
                total, Bank.N * 100);
    }
}