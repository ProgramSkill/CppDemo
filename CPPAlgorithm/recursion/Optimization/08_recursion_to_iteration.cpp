// recursion recursion_to_iteration
// Convert recursion to iteration using explicit stack
#include <iostream>
#include <stack>
using namespace std;

// [Recursive] Factorial
long long factorial_recursive(int n) {
    if (n <= 1) return 1;
    return n * factorial_recursive(n - 1);
}

// [Iterative] Factorial - simple loop
long long factorial_iterative(int n) {
    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// [Recursive] Sum of array
int sum_recursive(int arr[], int n) {
    if (n <= 0) return 0;
    return arr[n - 1] + sum_recursive(arr, n - 1);
}

// [Iterative] Sum of array
int sum_iterative(int arr[], int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += arr[i];
    }
    return result;
}

// [Recursive] Binary tree preorder traversal
struct Node {
    int data;
    Node* left;
    Node* right;
    Node(int val) : data(val), left(nullptr), right(nullptr) {}
};

void preorder_recursive(Node* node) {
    if (node == nullptr) return;
    cout << node->data << " ";
    preorder_recursive(node->left);
    preorder_recursive(node->right);
}

// [Iterative] Binary tree preorder using stack
void preorder_iterative(Node* root) {
    if (root == nullptr) return;

    stack<Node*> s;
    s.push(root);

    while (!s.empty()) {
        Node* node = s.top();
        s.pop();
        cout << node->data << " ";

        // Push right first so left is processed first
        if (node->right) s.push(node->right);
        if (node->left) s.push(node->left);
    }
}

int main() {
    cout << "=== RECURSION TO ITERATION ===" << endl << endl;

    // Factorial
    cout << "Factorial 5!:" << endl;
    cout << "Recursive: " << factorial_recursive(5) << endl;
    cout << "Iterative: " << factorial_iterative(5) << endl << endl;

    // Array sum
    int arr[] = {1, 2, 3, 4, 5};
    cout << "Sum of [1,2,3,4,5]:" << endl;
    cout << "Recursive: " << sum_recursive(arr, 5) << endl;
    cout << "Iterative: " << sum_iterative(arr, 5) << endl << endl;

    // Tree traversal
    Node* root = new Node(1);
    root->left = new Node(2);
    root->right = new Node(3);
    root->left->left = new Node(4);
    root->left->right = new Node(5);

    cout << "Tree preorder [1,2,4,5,3]:" << endl;
    cout << "Recursive: ";
    preorder_recursive(root);
    cout << endl;
    cout << "Iterative: ";
    preorder_iterative(root);
    cout << endl;

    // Cleanup
    delete root->left->left;
    delete root->left->right;
    delete root->left;
    delete root->right;
    delete root;

    return 0;
}

/*
CONVERSION PATTERNS:

1. Tail Recursion -> Simple Loop
   Recursive:                    Iterative:
   f(n, acc) {                   f(n) {
     if (n==0) return acc;         acc = initial;
     return f(n-1, g(acc));        while (n > 0) {
   }                                 acc = g(acc);
                                     n--;
                                   }
                                   return acc;
                                 }

2. Linear Recursion -> Loop with accumulator
   Just reverse the order of operations

3. Tree/Graph Recursion -> Explicit Stack
   Replace call stack with data structure stack
   Push children in reverse order

WHY CONVERT?
- Avoid stack overflow for deep recursion
- Better performance (no function call overhead)
- Required in languages without tail call optimization
*/
