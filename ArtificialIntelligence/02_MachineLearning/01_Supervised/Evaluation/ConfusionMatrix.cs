namespace ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

/// <summary>
/// 混淆矩阵（Confusion Matrix）
/// 用于可视化分类模型的性能
/// </summary>
public class ConfusionMatrix
{
    private int[,] _matrix;
    private int[] _classes;

    public ConfusionMatrix(int[] yTrue, int[] yPred)
    {
        if (yTrue.Length != yPred.Length)
            throw new ArgumentException("数组长度不匹配");

        _classes = yTrue.Concat(yPred).Distinct().OrderBy(x => x).ToArray();
        int numClasses = _classes.Length;
        _matrix = new int[numClasses, numClasses];

        // 构建混淆矩阵
        for (int i = 0; i < yTrue.Length; i++)
        {
            int trueIdx = Array.IndexOf(_classes, yTrue[i]);
            int predIdx = Array.IndexOf(_classes, yPred[i]);
            _matrix[trueIdx, predIdx]++;
        }
    }

    /// <summary>
    /// 获取混淆矩阵
    /// </summary>
    public int[,] GetMatrix() => (int[,])_matrix.Clone();

    /// <summary>
    /// 获取类别标签
    /// </summary>
    public int[] GetClasses() => (int[])_classes.Clone();

    /// <summary>
    /// 打印混淆矩阵
    /// </summary>
    public string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("混淆矩阵:");
        sb.Append("真实\\预测\t");

        foreach (var cls in _classes)
            sb.Append($"{cls}\t");
        sb.AppendLine();

        for (int i = 0; i < _classes.Length; i++)
        {
            sb.Append($"{_classes[i]}\t\t");
            for (int j = 0; j < _classes.Length; j++)
            {
                sb.Append($"{_matrix[i, j]}\t");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// 获取真正例数（True Positives）
    /// </summary>
    public int GetTruePositives(int classLabel)
    {
        int idx = Array.IndexOf(_classes, classLabel);
        return idx >= 0 ? _matrix[idx, idx] : 0;
    }

    /// <summary>
    /// 获取假正例数（False Positives）
    /// </summary>
    public int GetFalsePositives(int classLabel)
    {
        int idx = Array.IndexOf(_classes, classLabel);
        if (idx < 0) return 0;

        int sum = 0;
        for (int i = 0; i < _classes.Length; i++)
        {
            if (i != idx)
                sum += _matrix[i, idx];
        }
        return sum;
    }

    /// <summary>
    /// 获取假负例数（False Negatives）
    /// </summary>
    public int GetFalseNegatives(int classLabel)
    {
        int idx = Array.IndexOf(_classes, classLabel);
        if (idx < 0) return 0;

        int sum = 0;
        for (int j = 0; j < _classes.Length; j++)
        {
            if (j != idx)
                sum += _matrix[idx, j];
        }
        return sum;
    }

    /// <summary>
    /// 获取真负例数（True Negatives）
    /// </summary>
    public int GetTrueNegatives(int classLabel)
    {
        int idx = Array.IndexOf(_classes, classLabel);
        if (idx < 0) return 0;

        int sum = 0;
        for (int i = 0; i < _classes.Length; i++)
        {
            for (int j = 0; j < _classes.Length; j++)
            {
                if (i != idx && j != idx)
                    sum += _matrix[i, j];
            }
        }
        return sum;
    }
}
