// std::priority_queue - Max-heap (by default) container adapter
// Header: <queue>
// Time Complexity: O(log n) push, pop; O(1) top
// Space Complexity: O(N)
// Default container: std::vector with std::less comparator

#include <iostream>
#include <queue>
#include <vector>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::priority_queue Demo ===" << endl;
    priority_queue<int> pq;
    pq.push(3);
    pq.push(1);
    pq.push(4);
    pq.push(1);
    pq.push(5);
    cout << "Pushed: 3, 1, 4, 1, 5" << endl;
    cout << "Top (max): " << pq.top() << endl;
    cout << "Size: " << pq.size() << endl;
    pq.pop();
    cout << "After pop, top: " << pq.top() << endl;
    cout << endl;
}

void minHeapDemo() {
    cout << "=== Min-Heap (greater) ===" << endl;
    priority_queue<int, vector<int>, greater<int>> pq;
    pq.push(3);
    pq.push(1);
    pq.push(4);
    pq.push(5);
    cout << "Min-heap top: " << pq.top() << endl;
    cout << "Elements: ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl << endl;
}

void customComparatorDemo() {
    cout << "=== Custom Comparator ===" << endl;
    auto cmp = [](int a, int b) { return a > b; };  // Min-heap with lambda
    priority_queue<int, vector<int>, decltype(cmp)> pq(cmp);
    pq.push(10);
    pq.push(30);
    pq.push(20);
    cout << "Custom comparator (min-heap):" << endl;
    while (!pq.empty()) {
        cout << "  " << pq.top() << endl;
        pq.pop();
    }
    cout << endl;
}

void taskSchedulingDemo() {
    cout << "=== Use Case: Task Scheduling ===" << endl;
    struct Task {
        int priority;
        string name;
        bool operator<(const Task& other) const { return priority < other.priority; }
    };

    priority_queue<Task> tasks;
    tasks.push({3, "Low priority task"});
    tasks.push({1, "High priority task"});
    tasks.push({2, "Medium priority task"});

    cout << "Executing tasks by priority:" << endl;
    while (!tasks.empty()) {
        auto t = tasks.top(); pq.pop();
        cout << "  Priority " << t.priority << ": " << t.name << endl;
    }
    cout << endl;
}

void topKDemo() {
    cout << "=== Use Case: Top K Elements ===" << endl;
    vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
    int k = 3;

    // Using min-heap to find top k
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int num : nums) {
        pq.push(num);
        if (pq.size() > k) pq.pop();
    }

    cout << "Top " << k << " elements: ";
    while (!pq.empty()) {
        cout << pq.top() << " ";
        pq.pop();
    }
    cout << endl << endl;
}

void mergeKSortedDemo() {
    cout << "=== Use Case: Merge K Sorted Arrays ===" << endl;
    vector<vector<int>> arrays = {{1, 4, 5}, {1, 3, 4}, {2, 6}};

    using Element = pair<int, pair<int, int>>;  // value, (array index, element index)
    auto cmp = [](Element& a, Element& b) { return a.first > b.first; };
    priority_queue<Element, vector<Element>, decltype(cmp)> pq(cmp);

    for (int i = 0; i < arrays.size(); ++i) {
        if (!arrays[i].empty()) pq.push({arrays[i][0], {i, 0}});
    }

    cout << "Merged result: ";
    while (!pq.empty()) {
        auto [val, idx] = pq.top(); pq.pop();
        cout << val << " ";
        if (idx.second + 1 < arrays[idx.first].size()) {
            pq.push({arrays[idx.first][idx.second + 1], {idx.first, idx.second + 1}});
        }
    }
    cout << endl << endl;
}

void huffmanCodingDemo() {
    cout << "=== Use Case: Huffman Coding (simplified) ===" << endl;
    struct Node {
        int freq;
        char ch;
        bool operator<(const Node& other) const { return freq < other.freq; }
    };

    priority_queue<Node> pq;
    pq.push({5, 'A'});
    pq.push({9, 'B'});
    pq.push({12, 'C'});
    pq.push({13, 'D'});
    pq.push({16, 'E'});
    pq.push({45, 'F'});

    cout << "Processing by frequency (highest first):" << endl;
    while (!pq.empty()) {
        auto node = pq.top(); pq.pop();
        cout << "  '" << node.ch << "': " << node.freq << endl;
    }
    cout << endl;
}

void dijkstraDemo() {
    cout << "=== Use Case: Dijkstra's Algorithm (simplified) ===" << endl;
    using Edge = pair<int, int>;  // (neighbor, weight)
    vector<vector<Edge>> graph = {
        {{1, 4}, {2, 1}},  // Node 0
        {{3, 1}},          // Node 1
        {{1, 2}, {3, 5}},  // Node 2
        {}                 // Node 3
    };

    using PQElement = pair<int, int>;  // (distance, node)
    priority_queue<PQElement, vector<PQElement>, greater<PQElement>> pq;
    vector<int> dist(4, INT_MAX);

    dist[0] = 0;
    pq.push({0, 0});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;

        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }

    cout << "Shortest distances from node 0:" << endl;
    for (int i = 0; i < 4; ++i) {
        cout << "  Node " << i << ": " << dist[i] << endl;
    }
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "   std::priority_queue Demo\n";
    cout << "========================================\n\n";

    basicDemo();
    minHeapDemo();
    customComparatorDemo();
    taskSchedulingDemo();
    topKDemo();
    mergeKSortedDemo();
    huffmanCodingDemo();
    dijkstraDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::priority_queue: Heap adapter\n";
    cout << "  - Max-heap by default\n";
    cout << "  - O(log n) push/pop\n";
    cout << "  - O(1) top access\n";
    cout << "  - Use greater<> for min-heap\n";
    cout << "  - Perfect for priority queues, heaps\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
Top (max): 5
After pop, top: 4

=== Min-Heap ===
Top: 1
Elements: 1 1 3 4 5

=== Top K ===
Top 3 elements: 4 5 6
*/
