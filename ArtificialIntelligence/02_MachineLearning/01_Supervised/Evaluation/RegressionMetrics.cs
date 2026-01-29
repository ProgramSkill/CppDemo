namespace ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

/// <summary>
/// 回归模型评估指标
/// </summary>
public static class RegressionMetrics
{
    /// <summary>
    /// 均方误差（Mean Squared Error）
    /// </summary>
    public static double MeanSquaredError(double[] yTrue, double[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        double sum = 0;
        for (int i = 0; i < yTrue.Length; i++)
        {
            double error = yTrue[i] - yPred[i];
            sum += error * error;
        }

        return sum / yTrue.Length;
    }

    /// <summary>
    /// 均方根误差（Root Mean Squared Error）
    /// </summary>
    public static double RootMeanSquaredError(double[] yTrue, double[] yPred)
    {
        return Math.Sqrt(MeanSquaredError(yTrue, yPred));
    }

    /// <summary>
    /// 平均绝对误差（Mean Absolute Error）
    /// </summary>
    public static double MeanAbsoluteError(double[] yTrue, double[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        double sum = 0;
        for (int i = 0; i < yTrue.Length; i++)
        {
            sum += Math.Abs(yTrue[i] - yPred[i]);
        }

        return sum / yTrue.Length;
    }

    /// <summary>
    /// R²决定系数（R-squared）
    /// 表示模型解释的方差比例，值越接近1表示模型越好
    /// </summary>
    public static double RSquared(double[] yTrue, double[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        double mean = yTrue.Average();

        double ssTot = 0; // 总平方和
        double ssRes = 0; // 残差平方和

        for (int i = 0; i < yTrue.Length; i++)
        {
            ssTot += Math.Pow(yTrue[i] - mean, 2);
            ssRes += Math.Pow(yTrue[i] - yPred[i], 2);
        }

        return 1 - (ssRes / ssTot);
    }

    /// <summary>
    /// 平均绝对百分比误差（Mean Absolute Percentage Error）
    /// </summary>
    public static double MeanAbsolutePercentageError(double[] yTrue, double[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        double sum = 0;
        int count = 0;

        for (int i = 0; i < yTrue.Length; i++)
        {
            if (Math.Abs(yTrue[i]) > 1e-10) // 避免除以零
            {
                sum += Math.Abs((yTrue[i] - yPred[i]) / yTrue[i]);
                count++;
            }
        }

        return count > 0 ? (sum / count) * 100 : 0;
    }
}
