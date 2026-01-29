namespace ArtificialIntelligence.MachineLearning.Supervised.Classification;

/// <summary>
/// 高斯朴素贝叶斯分类器
/// 假设特征服从高斯分布，基于贝叶斯定理进行分类
/// </summary>
public class NaiveBayesClassifier
{
    private Dictionary<int, ClassStatistics>? _classStats;
    private Dictionary<int, double>? _classPriors;

    /// <summary>
    /// 训练模型
    /// </summary>
    public void Fit(double[,] X, int[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        var classes = y.Distinct().ToArray();
        _classStats = new Dictionary<int, ClassStatistics>();
        _classPriors = new Dictionary<int, double>();

        foreach (var cls in classes)
        {
            // 获取该类别的所有样本
            var classIndices = Enumerable.Range(0, n).Where(i => y[i] == cls).ToArray();
            int classCount = classIndices.Length;

            // 计算先验概率
            _classPriors[cls] = (double)classCount / n;

            // 计算每个特征的均值和标准差
            var means = new double[m];
            var stds = new double[m];

            for (int j = 0; j < m; j++)
            {
                var values = classIndices.Select(i => X[i, j]).ToArray();
                means[j] = values.Average();
                stds[j] = CalculateStandardDeviation(values, means[j]);
            }

            _classStats[cls] = new ClassStatistics { Means = means, Stds = stds };
        }
    }

    /// <summary>
    /// 预测
    /// </summary>
    public int[] Predict(double[,] X)
    {
        if (_classStats == null || _classPriors == null)
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
        int m = X.GetLength(1);
        double maxPosterior = double.NegativeInfinity;
        int bestClass = -1;

        foreach (var cls in _classStats!.Keys)
        {
            // 计算后验概率（使用对数避免下溢）
            double logPosterior = Math.Log(_classPriors![cls]);

            var stats = _classStats[cls];
            for (int j = 0; j < m; j++)
            {
                double value = X[index, j];
                double mean = stats.Means[j];
                double std = stats.Stds[j];

                // 高斯概率密度函数的对数
                logPosterior += LogGaussianPdf(value, mean, std);
            }

            if (logPosterior > maxPosterior)
            {
                maxPosterior = logPosterior;
                bestClass = cls;
            }
        }

        return bestClass;
    }

    private static double LogGaussianPdf(double x, double mean, double std)
    {
        if (std < 1e-10) std = 1e-10; // 避免除以零

        double exponent = -Math.Pow(x - mean, 2) / (2 * std * std);
        double logCoefficient = -Math.Log(std * Math.Sqrt(2 * Math.PI));

        return logCoefficient + exponent;
    }

    private static double CalculateStandardDeviation(double[] values, double mean)
    {
        if (values.Length <= 1) return 1e-10;

        double sumSquaredDiff = values.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquaredDiff / values.Length);
    }

    private class ClassStatistics
    {
        public double[] Means { get; set; } = Array.Empty<double>();
        public double[] Stds { get; set; } = Array.Empty<double>();
    }
}
