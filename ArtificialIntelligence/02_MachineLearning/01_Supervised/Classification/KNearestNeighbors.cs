namespace ArtificialIntelligence.MachineLearning.Supervised.Classification;

/// <summary>
/// K近邻分类器（K-Nearest Neighbors）
/// 基于实例的学习算法，通过找到K个最近邻居进行投票分类
/// </summary>
public class KNearestNeighbors
{
    private double[,]? _XTrain;
    private int[]? _yTrain;
    private int _k;

    /// <summary>
    /// 初始化KNN分类器
    /// </summary>
    /// <param name="k">近邻数量</param>
    public KNearestNeighbors(int k = 3)
    {
        if (k < 1)
            throw new ArgumentException("K必须大于0");
        _k = k;
    }

    /// <summary>
    /// 训练模型（存储训练数据）
    /// </summary>
    public void Fit(double[,] X, int[] y)
    {
        _XTrain = (double[,])X.Clone();
        _yTrain = (int[])y.Clone();
    }

    /// <summary>
    /// 预测
    /// </summary>
    public int[] Predict(double[,] X)
    {
        if (_XTrain == null || _yTrain == null)
            throw new InvalidOperationException("模型未训练");

        int n = X.GetLength(0);
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(X, i);
        }

        return predictions;
    }

    private int PredictSingle(double[,] X, int index)
    {
        int nTrain = _XTrain!.GetLength(0);

        // 计算到所有训练样本的距离
        var distances = new List<(double distance, int label)>();

        for (int i = 0; i < nTrain; i++)
        {
            double dist = EuclideanDistance(X, index, _XTrain, i);
            distances.Add((dist, _yTrain![i]));
        }

        // 找到K个最近邻
        var kNearest = distances.OrderBy(x => x.distance).Take(_k);

        // 投票决定类别
        var votes = kNearest.GroupBy(x => x.label)
                           .Select(g => new { Label = g.Key, Count = g.Count() })
                           .OrderByDescending(x => x.Count)
                           .First();

        return votes.Label;
    }

    private static double EuclideanDistance(double[,] X1, int idx1, double[,] X2, int idx2)
    {
        int m = X1.GetLength(1);
        double sum = 0.0;

        for (int j = 0; j < m; j++)
        {
            double diff = X1[idx1, j] - X2[idx2, j];
            sum += diff * diff;
        }

        return Math.Sqrt(sum);
    }
}
