namespace ArtificialIntelligence.MachineLearning.Supervised.Classification;

/// <summary>
/// 决策树分类器
/// 使用信息增益（基于熵）进行特征选择和分裂
/// </summary>
public class DecisionTreeClassifier
{
    private TreeNode? _root;
    private int _maxDepth;
    private int _minSamplesSplit;

    public DecisionTreeClassifier(int maxDepth = 10, int minSamplesSplit = 2)
    {
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
    }

    /// <summary>
    /// 训练决策树
    /// </summary>
    public void Fit(double[,] X, int[] y)
    {
        _root = BuildTree(X, y, 0);
    }

    /// <summary>
    /// 预测
    /// </summary>
    public int[] Predict(double[,] X)
    {
        if (_root == null)
            throw new InvalidOperationException("模型未训练");

        int n = X.GetLength(0);
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(X, i, _root);
        }

        return predictions;
    }

    private TreeNode BuildTree(double[,] X, int[] y, int depth)
    {
        int n = y.Length;
        int numClasses = y.Distinct().Count();

        // 停止条件
        if (depth >= _maxDepth || n < _minSamplesSplit || numClasses == 1)
        {
            return new TreeNode { IsLeaf = true, PredictedClass = MostCommonLabel(y) };
        }

        // 找到最佳分裂
        var (featureIdx, threshold) = FindBestSplit(X, y);

        if (featureIdx == -1)
        {
            return new TreeNode { IsLeaf = true, PredictedClass = MostCommonLabel(y) };
        }

        // 分裂数据
        var (leftIndices, rightIndices) = Split(X, featureIdx, threshold);

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            return new TreeNode { IsLeaf = true, PredictedClass = MostCommonLabel(y) };
        }

        // 递归构建子树
        var leftX = GetSubset(X, leftIndices);
        var leftY = GetSubset(y, leftIndices);
        var rightX = GetSubset(X, rightIndices);
        var rightY = GetSubset(y, rightIndices);

        return new TreeNode
        {
            IsLeaf = false,
            FeatureIndex = featureIdx,
            Threshold = threshold,
            Left = BuildTree(leftX, leftY, depth + 1),
            Right = BuildTree(rightX, rightY, depth + 1)
        };
    }

    private (int featureIdx, double threshold) FindBestSplit(double[,] X, int[] y)
    {
        int m = X.GetLength(1);
        double bestGain = -1;
        int bestFeature = -1;
        double bestThreshold = 0;

        double parentEntropy = CalculateEntropy(y);

        for (int featureIdx = 0; featureIdx < m; featureIdx++)
        {
            var thresholds = GetUniqueValues(X, featureIdx);

            foreach (var threshold in thresholds)
            {
                var (leftIndices, rightIndices) = Split(X, featureIdx, threshold);

                if (leftIndices.Count == 0 || rightIndices.Count == 0)
                    continue;

                var leftY = GetSubset(y, leftIndices);
                var rightY = GetSubset(y, rightIndices);

                double gain = InformationGain(y, leftY, rightY, parentEntropy);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = featureIdx;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeature, bestThreshold);
    }

    private static double InformationGain(int[] parent, int[] left, int[] right, double parentEntropy)
    {
        double n = parent.Length;
        double nLeft = left.Length;
        double nRight = right.Length;

        double childEntropy = (nLeft / n) * CalculateEntropy(left) + (nRight / n) * CalculateEntropy(right);
        return parentEntropy - childEntropy;
    }

    private static double CalculateEntropy(int[] y)
    {
        var counts = y.GroupBy(x => x).Select(g => g.Count()).ToArray();
        double n = y.Length;
        double entropy = 0;

        foreach (var count in counts)
        {
            double p = count / n;
            if (p > 0)
                entropy -= p * Math.Log2(p);
        }

        return entropy;
    }

    private static (List<int> left, List<int> right) Split(double[,] X, int featureIdx, double threshold)
    {
        var left = new List<int>();
        var right = new List<int>();

        for (int i = 0; i < X.GetLength(0); i++)
        {
            if (X[i, featureIdx] <= threshold)
                left.Add(i);
            else
                right.Add(i);
        }

        return (left, right);
    }

    private static double[] GetUniqueValues(double[,] X, int featureIdx)
    {
        var values = new HashSet<double>();
        for (int i = 0; i < X.GetLength(0); i++)
        {
            values.Add(X[i, featureIdx]);
        }
        return values.OrderBy(x => x).ToArray();
    }

    private static double[,] GetSubset(double[,] X, List<int> indices)
    {
        int m = X.GetLength(1);
        var subset = new double[indices.Count, m];

        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < m; j++)
            {
                subset[i, j] = X[indices[i], j];
            }
        }

        return subset;
    }

    private static int[] GetSubset(int[] y, List<int> indices)
    {
        return indices.Select(i => y[i]).ToArray();
    }

    private static int MostCommonLabel(int[] y)
    {
        return y.GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key;
    }

    private int PredictSingle(double[,] X, int index, TreeNode node)
    {
        if (node.IsLeaf)
            return node.PredictedClass;

        if (X[index, node.FeatureIndex] <= node.Threshold)
            return PredictSingle(X, index, node.Left!);
        else
            return PredictSingle(X, index, node.Right!);
    }

    private class TreeNode
    {
        public bool IsLeaf { get; set; }
        public int PredictedClass { get; set; }
        public int FeatureIndex { get; set; }
        public double Threshold { get; set; }
        public TreeNode? Left { get; set; }
        public TreeNode? Right { get; set; }
    }
}
