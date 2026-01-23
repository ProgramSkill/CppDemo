// std::stack - LIFO (Last In First Out) container adapter
// Header: <stack>
// Time Complexity: O(1) push, pop, top
// Space Complexity: O(N)
// Default container: std::deque

#include <iostream>
#include <stack>
#include <vector>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::stack Demo ===" << endl;
    stack<int> s;
    s.push(1);
    s.push(2);
    s.push(3);
    cout << "Pushed 1, 2, 3" << endl;
    cout << "Top: " << s.top() << endl;
    cout << "Size: " << s.size() << endl;
    s.pop();
    cout << "After pop, top: " << s.top() << endl;
    cout << endl;
}

void operationsDemo() {
    cout << "=== Stack Operations ===" << endl;
    stack<string> s;
    s.push("first");
    s.emplace("second");  // C++11
    cout << "After push and emplace:" << endl;
    cout << "  Size: " << s.size() << ", Top: " << s.top() << endl;
    s.pop();
    cout << "  After pop, top: " << s.top() << endl;
    cout << "  Empty: " << boolalpha << s.empty() << endl;
    cout << endl;
}

void customContainerDemo() {
    cout << "=== Custom Container ===" << endl;
    stack<int, vector<int>> s;  // Using vector as underlying container
    s.push(1);
    s.push(2);
    s.push(3);
    cout << "Stack using vector:" << endl;
    while (!s.empty()) {
        cout << "  " << s.top() << endl;
        s.pop();
    }
    cout << endl;
}

void palindromeDemo() {
    cout << "=== Use Case: Palindrome Check ===" << endl;
    string str = "racecar";
    stack<char> s;
    for (char c : str) s.push(c);
    string reversed;
    while (!s.empty()) {
        reversed += s.top();
        s.pop();
    }
    cout << "Original: " << str << endl;
    cout << "Reversed: " << reversed << endl;
    cout << "Is palindrome: " << boolalpha << (str == reversed) << endl;
    cout << endl;
}

void undoRedoDemo() {
    cout << "=== Use Case: Undo/Redo ===" << endl;
    stack<string> undoStack;
    stack<string> redoStack;

    auto action = [&](const string& op) {
        if (op == "write") undoStack.push(op);
        else if (op == "undo" && !undoStack.empty()) {
            redoStack.push(undoStack.top());
            undoStack.pop();
            cout << "  Undone: " << redoStack.top() << endl;
        }
        else if (op == "redo" && !redoStack.empty()) {
            undoStack.push(redoStack.top());
            redoStack.pop();
            cout << "  Redone: " << undoStack.top() << endl;
        }
    };

    action("write");
    action("write");
    action("undo");
    action("redo");
    action("undo");
    cout << endl;
}

void dfsDemo() {
    cout << "=== Use Case: DFS Traversal ===" << endl;
    vector<vector<int>> graph = {{1, 2}, {0, 2}, {0, 1, 3}, {2}};
    stack<int> s;
    vector<bool> visited(4, false);

    s.push(0);
    cout << "DFS starting from node 0: ";
    while (!s.empty()) {
        int node = s.top(); s.pop();
        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";
            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) s.push(neighbor);
            }
        }
    }
    cout << endl << endl;
}

void parenthesisMatchingDemo() {
    cout << "=== Use Case: Parenthesis Matching ===" << endl;
    string expr = "{[()]}";
    stack<char> s;
    bool valid = true;

    for (char c : expr) {
        if (c == '{' || c == '[' || c == '(') {
            s.push(c);
        } else if (!s.empty()) {
            char top = s.top(); s.pop();
            if ((c == '}' && top != '{') ||
                (c == ']' && top != '[') ||
                (c == ')' && top != '(')) {
                valid = false;
                break;
            }
        }
    }
    cout << "Expression: " << expr << endl;
    cout << "Valid: " << boolalpha << (valid && s.empty()) << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "      std::stack Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    operationsDemo();
    customContainerDemo();
    palindromeDemo();
    undoRedoDemo();
    dfsDemo();
    parenthesisMatchingDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::stack: LIFO container adapter\n";
    cout << "  - push/emplace: O(1)\n";
    cout << "  - pop: O(1)\n";
    cout << "  - top: O(1)\n";
    cout << "  - No iteration!\n";
    cout << "  - Default: deque, can use vector/list\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
Top: 3
After pop, top: 2

=== Palindrome ===
Original: racecar
Reversed: racecar
Is palindrome: true
*/
