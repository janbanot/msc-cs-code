package Lab4;

import java.util.*;
import java.util.concurrent.*;

class Internet { // Przykładowy "Internet"
    private static final Map<String, String> urlContentMap = new HashMap<>();

    static {
        urlContentMap.put("http://example.com",
                "http://example.com/page1\nhttp://example.com/page2");
        urlContentMap.put("http://example.com/page1",
                "http://example.com/page3\nhttp://example.com/page4");
        urlContentMap.put("http://example.com/page2",
                "http://example.com/page5");
        urlContentMap.put("http://example.com/page3", "");
        urlContentMap.put("http://example.com/page4",
                "http://example.com/page6\nhttp://example.org/page1");
        urlContentMap.put("http://example.com/page5",
                "http://example.com/page6\nhttp://example.com/page7\nhttp://example.net/page1");
        urlContentMap.put("http://example.com/page6", "");
        urlContentMap.put("http://example.com/page7", "");
        urlContentMap.put("http://example.org/page1",
                "http://example.org/page2\nhttp://example.com/page3");
        urlContentMap.put("http://example.org/page2", "");
        urlContentMap.put("http://example.net/page1",
                "http://example.net/page2");
        urlContentMap.put("http://example.net/page2", "");
    }

    public static String get(String url) {
        try {
            Thread.sleep(ThreadLocalRandom.current().nextInt(1000));
        } catch (InterruptedException _) {
        }
        return urlContentMap.getOrDefault(url, null);
    }

    public static Set<String> getAllUrls() {
        return urlContentMap.keySet();
    }
}


class Task {
    private final String url;
    private final int depth;

    public Task(String url, int depth) {
        this.url = url;
        this.depth = depth;
    }

    public String getUrl() {
        return url;
    }

    public int getDepth() {
        return depth;
    }
}

class WebCrawlerThread extends Thread {
    private final BlockingQueue<Task> queue;
    private final Set<String> visited;
    private final int maxDepth;

    public WebCrawlerThread(BlockingQueue<Task> queue, Set<String> visited, int maxDepth) {
        this.queue = queue;
        this.visited = visited;
        this.maxDepth = maxDepth;
    }

    @Override
    public void run() {
        while (true) {
            try {
                // Wait for a task from the queue, timeout after 2 seconds
                Task task = queue.poll(2, TimeUnit.SECONDS);
                if (task == null) {
                    // Exit if no tasks are available
                    break;
                }

                String url = task.getUrl();
                int depth = task.getDepth();

                // Skip if we've reached max depth
                if (depth > maxDepth) {
                    continue;
                }

                // Skip if URL has already been visited
                synchronized (visited) {
                    if (visited.contains(url)) {
                        continue;
                    }
                    visited.add(url);
                }

                System.out.println("Crawled URL: " + url + " at depth " + depth);
                String content = Internet.get(url);

                if (content != null) {
                    // Extract new URLs from the content
                    String[] newUrls = content.split("\n");
                    for (String newUrl : newUrls) {
                        // Skip empty URLs
                        if (newUrl.trim().isEmpty()) {
                            continue;
                        }

                        synchronized (visited) {
                            if (!visited.contains(newUrl)) {
                                // Add new task to the queue
                                queue.put(new Task(newUrl, depth + 1));
                            }
                        }
                    }
                }
            } catch (InterruptedException e) {
                // Handle thread interruption
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}


public class WebCrawler {
    private final BlockingQueue<Task> queue = new LinkedBlockingQueue<>();
    private final Set<String> visited = new HashSet<>();
    private final int maxDepth;
    private final int numThreads;

    public WebCrawler(int maxDepth, int numThreads) {
        this.maxDepth = maxDepth;
        this.numThreads = numThreads;
    }

    public void startCrawling(String startUrl) throws InterruptedException {
        queue.put(new Task(startUrl, 0));
        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < numThreads; i++) {
            threads[i] = new WebCrawlerThread(queue, visited, maxDepth);
            threads[i].start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
    }

    public Set<String> getVisitedUrls() {
        return visited;
    }

    public static void main(String[] args) throws InterruptedException {
        WebCrawler webCrawler = new WebCrawler(3, 4);
        webCrawler.startCrawling("http://example.com");
        // Weryfikacja
        Set<String> visitedUrls = webCrawler.getVisitedUrls();
        Set<String> allUrls = Internet.getAllUrls();
        if (visitedUrls.containsAll(allUrls)) {
            System.out.println("All pages have been successfully crawled.");
        } else {
            Set<String> missedUrls = new HashSet<>(allUrls);
            missedUrls.removeAll(visitedUrls);
            System.out.println("The following pages were not crawled: " + missedUrls);
        }
    }
}