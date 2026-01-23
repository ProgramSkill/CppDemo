#include <iostream>
#include <queue>
#include <deque>
#include <list>
#include <vector>
#include <string>

// ============================================================================
// std::queue 详细解析
// ============================================================================
// std::queue 是一个容器适配器（container adapter），它提供先进先出（FIFO）的数据结构
//
// 特性：
// 1. 只允许在队尾（back）插入元素
// 2. 只允许在队头（front）删除和访问元素
// 3. 不支持迭代器
// 4. 默认使用 std::deque 作为底层容器
// 5. 可以使用 std::list 或其他满足 SequenceContainer 要求的容器
//
// 时间复杂度：
// - push/emplace: O(1)
// - pop: O(1)
// - front: O(1)
// - back: O(1)
// - size: O(1)
// - empty: O(1)
// ============================================================================

void basicOperations() {
    std::cout << "========== 基本操作 ==========" << std::endl;

    // 1. 创建 queue
    std::queue<int> q;  // 默认使用 deque

    // 2. 插入元素 (push/emplace)
    q.push(10);
    q.push(20);
    q.push(30);
    q.emplace(40);  // emplace 就地构造，避免临时对象

    std::cout << "队列元素: ";
    std::queue<int> temp = q;
    while (!temp.empty()) {
        std::cout << temp.front() << " ";
        temp.pop();
    }
    std::cout << std::endl;

    // 3. 访问元素
    std::cout << "队头元素 (front): " << q.front() << std::endl;  // 10
    std::cout << "队尾元素 (back): " << q.back() << std::endl;    // 40

    // 4. 删除元素
    q.pop();  // 删除队头元素 (10)
    std::cout << "删除队头后，新的队头: " << q.front() << std::endl;

    // 5. 大小和判断
    std::cout << "队列大小 (size): " << q.size() << std::endl;
    std::cout << "队列是否为空 (empty): " << (q.empty() ? "是" : "否") << std::endl;
}

void queueWithDifferentContainers() {
    std::cout << "\n========== 使用不同底层容器 ==========" << std::endl;

    // 使用 deque (默认)
    std::queue<int, std::deque<int>> qDeque;
    qDeque.push(1);
    qDeque.push(2);
    std::cout << "queue<deque>: front=" << qDeque.front() << ", size=" << qDeque.size() << std::endl;

    // 使用 list
    std::queue<int, std::list<int>> qList;
    qList.push(1);
    qList.push(2);
    std::cout << "queue<list>: front=" << qList.front() << ", size=" << qList.size() << std::endl;

    // 注意：不能使用 vector 作为 queue 的底层容器
    // 因为 vector 不提供 pop_front() 成员函数
    // std::queue<int, std::vector<int>> qVector;  // 编译错误！
}

void queueWithCustomTypes() {
    std::cout << "\n========== 自定义类型队列 ==========" << std::endl;

    struct Task {
        int id;
        std::string name;
        int priority;

        Task(int i, const std::string& n, int p) : id(i), name(n), priority(p) {}
    };

    std::queue<Task> taskQueue;

    // 插入任务
    taskQueue.emplace(1, "处理数据", 5);
    taskQueue.emplace(2, "发送邮件", 3);
    taskQueue.emplace(3, "生成报告", 7);

    // 处理任务
    while (!taskQueue.empty()) {
        Task task = taskQueue.front();
        std::cout << "执行任务 [ID:" << task.id << "] "
                  << task.name << " (优先级:" << task.priority << ")" << std::endl;
        taskQueue.pop();
    }
}

void bfsExample() {
    std::cout << "\n========== 广度优先搜索 (BFS) 示例 ==========" << std::endl;

    // 简单的图表示 (邻接表)
    std::vector<std::vector<int>> graph = {
        {1, 2},      // 节点 0 的邻居
        {0, 3, 4},   // 节点 1 的邻居
        {0, 4},      // 节点 2 的邻居
        {1},         // 节点 3 的邻居
        {1, 2}       // 节点 4 的邻居
    };

    int startNode = 0;
    std::queue<int> bfsQueue;
    std::vector<bool> visited(graph.size(), false);

    // BFS 遍历
    bfsQueue.push(startNode);
    visited[startNode] = true;

    std::cout << "BFS 遍历顺序: ";
    while (!bfsQueue.empty()) {
        int current = bfsQueue.front();
        bfsQueue.pop();

        std::cout << current << " ";

        // 访问所有未访问的邻居
        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                bfsQueue.push(neighbor);
            }
        }
    }
    std::cout << std::endl;
}

void taskScheduler() {
    std::cout << "\n========== 任务调度器示例 ==========" << std::endl;

    std::queue<std::string> printQueue;

    // 添加打印任务
    printQueue.push("文档1.pdf");
    printQueue.push("文档2.docx");
    printQueue.push("文档3.xlsx");

    // 模拟打印队列
    int jobId = 1;
    while (!printQueue.empty()) {
        std::string document = printQueue.front();
        std::cout << "[作业 #" << jobId++ << "] 正在打印: " << document << std::endl;
        printQueue.pop();
    }
    std::cout << "所有打印任务已完成" << std::endl;
}

void levelOrderTraversal() {
    std::cout << "\n========== 二叉树层序遍历示例 ==========" << std::endl;

    struct TreeNode {
        int val;
        TreeNode* left;
        TreeNode* right;
        TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    };

    // 构建简单二叉树
    //       1
    //      / \
    //     2   3
    //    / \
    //   4   5
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    root->left->left = new TreeNode(4);
    root->left->right = new TreeNode(5);

    // 层序遍历
    std::queue<TreeNode*> levelQueue;
    levelQueue.push(root);

    std::cout << "层序遍历结果: ";
    while (!levelQueue.empty()) {
        TreeNode* node = levelQueue.front();
        levelQueue.pop();

        std::cout << node->val << " ";

        if (node->left) levelQueue.push(node->left);
        if (node->right) levelQueue.push(node->right);
    }
    std::cout << std::endl;

    // 清理内存
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;
}

void swappingQueues() {
    std::cout << "\n========== 交换队列内容 ==========" << std::endl;

    std::queue<int> q1;
    q1.push(1);
    q1.push(2);
    q1.push(3);

    std::queue<int> q2;
    q2.push(100);
    q2.push(200);

    std::cout << "交换前 q1 大小: " << q1.size() << ", q2 大小: " << q2.size() << std::endl;

    // 交换两个队列的内容
    q1.swap(q2);

    std::cout << "交换后 q1 大小: " << q1.size() << ", q2 大小: " << q2.size() << std::endl;
    std::cout << "q1.front(): " << q1.front() << ", q2.front(): " << q2.front() << std::endl;
}

void queueComparison() {
    std::cout << "\n========== queue 的比较运算 ==========" << std::endl;

    std::queue<int> q1, q2;

    q1.push(1);
    q1.push(2);
    q1.push(3);

    q2.push(1);
    q2.push(2);
    q2.push(3);

    // 比较运算符需要相同类型和相同的底层容器类型
    std::cout << "q1 == q2: " << (q1 == q2 ? "相等" : "不相等") << std::endl;

    q2.pop();
    std::cout << "q2 移除一个元素后" << std::endl;
    std::cout << "q1 == q2: " << (q1 == q2 ? "相等" : "不相等") << std::endl;
    std::cout << "q1 != q2: " << (q1 != q2 ? "不相等" : "相等") << std::endl;
}

void moveSemantics() {
    std::cout << "\n========== 移动语义示例 (C++11) ==========" << std::endl;

    std::queue<std::string> q1;
    q1.push("Hello");
    q1.push("World");

    std::cout << "q1 大小: " << q1.size() << std::endl;

    // 移动构造
    std::queue<std::string> q2 = std::move(q1);

    std::cout << "移动后 q1 大小: " << q1.size() << " (已移动)" << std::endl;
    std::cout << "移动后 q2 大小: " << q2.size() << std::endl;

    // 移动赋值
    std::queue<std::string> q3;
    q3 = std::move(q2);

    std::cout << "再次移动后 q2 大小: " << q2.size() << " (已移动)" << std::endl;
    std::cout << "再次移动后 q3 大小: " << q3.size() << std::endl;
}

void performanceNotes() {
    std::cout << "\n========== 性能考虑 ==========" << std::endl;
    std::cout << "1. queue 的所有操作都是 O(1) 时间复杂度" << std::endl;
    std::cout << "2. deque 作为默认容器提供最佳性能平衡" << std::endl;
    std::cout << "3. list 适合频繁插入删除的场景" << std::endl;
    std::cout << "4. queue 不提供迭代器，不能遍历访问" << std::endl;
    std::cout << "5. queue 不支持随机访问元素" << std::endl;
    std::cout << "6. emplace 比 push 更高效（避免临时对象）" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "       std::queue 详细解析演示" << std::endl;
    std::cout << "========================================" << std::endl;

    basicOperations();
    queueWithDifferentContainers();
    queueWithCustomTypes();
    bfsExample();
    taskScheduler();
    levelOrderTraversal();
    swappingQueues();
    queueComparison();
    moveSemantics();
    performanceNotes();

    std::cout << "\n========================================" << std::endl;
    std::cout << "           演示结束" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
