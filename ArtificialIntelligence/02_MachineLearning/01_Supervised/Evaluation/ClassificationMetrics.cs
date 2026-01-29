namespace ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

/// <summary>
/// 分类模型评估指标
/// </summary>
public static class ClassificationMetrics
{
    /// <summary>
    /// 准确率（Accuracy）
    /// </summary>
    public static double Accuracy(int[] yTrue, int[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        int correct = 0;
        for (int i = 0; i < yTrue.Length; i++)
        {
            if (yTrue[i] == yPred[i])
                correct++;
        }

        return (double)correct / yTrue.Length;
    }

    /// <summary>
    /// 精确率（Precision）- 针对二分类
    /// </summary>
    public static double Precision(int[] yTrue, int[] yPred, int positiveClass = 1)
    {
        int truePositive = 0;
        int falsePositive = 0;

        for (int i = 0; i < yTrue.Length; i++)
        {
            if (yPred[i] == positiveClass)
            {
                if (yTrue[i] == positiveClass)
                    truePositive++;
                else
                    falsePositive++;
            }
        }

        return truePositive + falsePositive > 0
            ? (double)truePositive / (truePositive + falsePositive)
            : 0;
    }

    /// <summary>
    /// 召回率（Recall）- 针对二分类
    /// </summary>
    public static double Recall(int[] yTrue, int[] yPred, int positiveClass = 1)
    {
        int truePositive = 0;
        int falseNegative = 0;

        for (int i = 0; i < yTrue.Length; i++)
        {
            if (yTrue[i] == positiveClass)
            {
                if (yPred[i] == positiveClass)
                    truePositive++;
                else
                    falseNegative++;
            }
        }

        return truePositive + falseNegative > 0
            ? (double)truePositive / (truePositive + falseNegative)
            : 0;
    }

    /// <summary>
    /// F1分数（F1-Score）- 精确率和召回率的调和平均
    /// </summary>
    public static double F1Score(int[] yTrue, int[] yPred, int positiveClass = 1)
    {
        double precision = Precision(yTrue, yPred, positiveClass);
        double recall = Recall(yTrue, yPred, positiveClass);

        return precision + recall > 0
            ? 2 * (precision * recall) / (precision + recall)
            : 0;
    }

    /// <summary>
    /// 特异度（Specificity）- 真负例率
    /// </summary>
    public static double Specificity(int[] yTrue, int[] yPred, int positiveClass = 1)
    {
        int trueNegative = 0;
        int falsePositive = 0;

        for (int i = 0; i < yTrue.Length; i++)
        {
            if (yTrue[i] != positiveClass)
            {
                if (yPred[i] != positiveClass)
                    trueNegative++;
                else
                    falsePositive++;
            }
        }

        return trueNegative + falsePositive > 0
            ? (double)trueNegative / (trueNegative + falsePositive)
            : 0;
    }
}
