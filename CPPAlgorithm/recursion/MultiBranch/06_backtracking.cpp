// recursion backtracking
// Backtracking: Try all possibilities, undo choices that don't work
#include <iostream>
#include <vector>
using namespace std;

// Print current permutation
void printVector(const vector<int>& v) {
    cout << "[ ";
    for (int num : v) {
        cout << num << " ";
    }
    cout << "]" << endl;
}

// Generate all permutations of numbers 1 to n
void permute(vector<int>& current, vector<bool>& used, int n) {
    // [Base Case] Found a complete permutation
    if (current.size() == n) {
        printVector(current);
        return;
    }

    // [Recursive Step] Try each unused number
    for (int i = 1; i <= n; i++) {
        if (used[i]) continue;  // Skip if already used

        // Make choice
        current.push_back(i);
        used[i] = true;
        cout << "Choose " << i << ", current: ";
        printVector(current);

        // Recurse
        permute(current, used, n);

        // Backtrack (undo choice)
        current.pop_back();
        used[i] = false;
        cout << "Backtrack, remove " << i << endl;
    }
}

int main() {
    cout << "=== BACKTRACKING: PERMUTATIONS ===" << endl;
    cout << "Generate all permutations of [1, 2, 3]" << endl << endl;

    int n = 3;
    vector<int> current;
    vector<bool> used(n + 1, false);

    permute(current, used, n);

    cout << "\n=== ALL PERMUTATIONS ===" << endl;
    cout << "Total: " << n << "! = 6 permutations" << endl;

    return 0;
}

/*
BACKTRACKING PATTERN:

void backtrack(state) {
    if (is_solution(state)) {
        process_solution(state);
        return;
    }

    for (choice in choices) {
        if (is_valid(choice)) {
            make_choice(choice);      // Modify state
            backtrack(state);         // Recurse
            undo_choice(choice);      // Restore state (BACKTRACK!)
        }
    }
}

PERMUTATION TREE for n=3:

                        []
           /            |            \
         [1]           [2]           [3]
        /   \         /   \         /   \
     [1,2] [1,3]   [2,1] [2,3]   [3,1] [3,2]
       |     |       |     |       |     |
   [1,2,3][1,3,2] [2,1,3][2,3,1] [3,1,2][3,2,1]

Common backtracking problems:
- N-Queens
- Sudoku solver
- Subset sum
- Maze solving
- Graph coloring
*/
