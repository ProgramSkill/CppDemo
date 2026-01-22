// recursion tree_traversal
#include <iostream>
#include <queue>
using namespace std;

// Define binary tree node structure
struct Node {
    int data;
    Node* left;
    Node* right;

    // Constructor for easy node creation
    Node(int value) : data(value), left(nullptr), right(nullptr) {}
};

// Class to manage tree operations
class BinaryTree {
private:
    Node* root;

    // Helper function to recursively delete all nodes
    void destroyTree(Node* node) {
        if (node == nullptr) return;
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }

public:
    BinaryTree() : root(nullptr) {}

    // Destructor to free all allocated memory
    ~BinaryTree() {
        destroyTree(root);
    }

    // Create a sample tree for demonstration
    void createSampleTree() {
        root = new Node(1);
        root->left = new Node(2);
        root->right = new Node(3);
        root->left->left = new Node(4);
        root->left->right = new Node(5);
        root->right->left = new Node(6);
        root->right->right = new Node(7);

        cout << "Sample tree created:" << endl;
        cout << "        1" << endl;
        cout << "       / \\" << endl;
        cout << "      2   3" << endl;
        cout << "     / \\ / \\" << endl;
        cout << "    4  5 6  7" << endl;
        cout << endl;
    }

    // [Traversal 1] Preorder: Root -> Left -> Right
    // Use case: Making a copy of the tree
    void preorder(Node* node) {
        if (node == nullptr) {
            return;  // Base case: reach leaf node's child
        }

        // Process root first
        cout << node->data << " ";

        // Then traverse left subtree
        preorder(node->left);

        // Finally traverse right subtree
        preorder(node->right);
    }

    // [Traversal 2] Inorder: Left -> Root -> Right
    // Use case: Get sorted output from BST (Binary Search Tree)
    void inorder(Node* node) {
        if (node == nullptr) {
            return;  // Base case
        }

        // Traverse left subtree first
        inorder(node->left);

        // Then process root
        cout << node->data << " ";

        // Finally traverse right subtree
        inorder(node->right);
    }

    // [Traversal 3] Postorder: Left -> Right -> Root
    // Use case: Deleting tree, calculating tree height
    void postorder(Node* node) {
        if (node == nullptr) {
            return;  // Base case
        }

        // Traverse left subtree first
        postorder(node->left);

        // Then traverse right subtree
        postorder(node->right);

        // Process root last
        cout << node->data << " ";
    }

    // [Traversal 4] Level Order: Top to Bottom, Left to Right
    // Use case: Finding shortest path, BFS
    void levelorder(Node* node) {
        if (node == nullptr) return;

        queue<Node*> q;
        q.push(node);

        while (!q.empty()) {
            Node* current = q.front();
            q.pop();

            cout << current->data << " ";

            if (current->left != nullptr) {
                q.push(current->left);
            }
            if (current->right != nullptr) {
                q.push(current->right);
            }
        }
    }

    // Calculate tree height using recursion
    // Height = max(height of left subtree, height of right subtree) + 1
    int getHeight(Node* node) {
        if (node == nullptr) {
            return 0;  // Base case: empty tree has height 0
        }

        int leftHeight = getHeight(node->left);
        int rightHeight = getHeight(node->right);

        return max(leftHeight, rightHeight) + 1;
    }

    // Count total nodes in tree using recursion
    // Total nodes = 1 + nodes in left subtree + nodes in right subtree
    int countNodes(Node* node) {
        if (node == nullptr) {
            return 0;  // Base case: empty tree has 0 nodes
        }

        return 1 + countNodes(node->left) + countNodes(node->right);
    }

    // Search for a value in tree using recursion
    bool search(Node* node, int target) {
        if (node == nullptr) {
            return false;  // Base case: not found
        }

        if (node->data == target) {
            return true;  // Base case: found!
        }

        // Search in left or right subtree
        return search(node->left, target) || search(node->right, target);
    }

    // Public wrapper functions
    void printPreorder() {
        cout << "Preorder (Root -> Left -> Right): ";
        preorder(root);
        cout << endl;
    }

    void printInorder() {
        cout << "Inorder (Left -> Root -> Right): ";
        inorder(root);
        cout << endl;
    }

    void printPostorder() {
        cout << "Postorder (Left -> Right -> Root): ";
        postorder(root);
        cout << endl;
    }

    void printLevelorder() {
        cout << "Level Order (Top to Bottom): ";
        levelorder(root);
        cout << endl;
    }

    void showTreeInfo() {
        cout << "\nTree Information:" << endl;
        cout << "Height: " << getHeight(root) << endl;
        cout << "Total Nodes: " << countNodes(root) << endl;
    }

    void searchValue(int target) {
        cout << "\nSearching for " << target << ": ";
        if (search(root, target)) {
            cout << "Found!" << endl;
        }
        else {
            cout << "Not found!" << endl;
        }
    }
};

int main() {
    BinaryTree tree;
    tree.createSampleTree();

    cout << "=== TREE TRAVERSAL METHODS ===" << endl << endl;

    tree.printPreorder();
    cout << "Explanation: Visit root, then left subtree, then right subtree" << endl << endl;

    tree.printInorder();
    cout << "Explanation: Visit left subtree, then root, then right subtree" << endl << endl;

    tree.printPostorder();
    cout << "Explanation: Visit left subtree, then right subtree, then root" << endl << endl;

    tree.printLevelorder();
    cout << "Explanation: Visit nodes level by level, left to right" << endl << endl;

    tree.showTreeInfo();

    tree.searchValue(5);
    tree.searchValue(10);

    return 0;
}

/*
TREE TRAVERSAL VISUALIZATION:

Tree structure:
        1
       / \
      2   3
     / \ / \
    4  5 6  7

Preorder (Root-Left-Right):
1 -> 2 -> 4 -> 5 -> 3 -> 6 -> 7
Process root FIRST, then children

Inorder (Left-Root-Right):
4 -> 2 -> 5 -> 1 -> 6 -> 3 -> 7
Process root in MIDDLE

Postorder (Left-Right-Root):
4 -> 5 -> 2 -> 6 -> 7 -> 3 -> 1
Process root LAST (opposite of preorder)

Level Order (BFS):
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
Level 1: 1
Level 2: 2, 3
Level 3: 4, 5, 6, 7
*/