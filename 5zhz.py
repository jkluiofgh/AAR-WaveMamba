import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import os
import argparse
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str,  help='saved path of each setting')
    args = parser.parse_args()
    checkpoint_path = args.save_path 

    test_target_1 = torch.load(os.path.join(checkpoint_path,"1",'test target.pt'))
    test_predict_1 = torch.load(os.path.join(checkpoint_path,"1",'test predict.pt'))

    test_target_2 = torch.load(os.path.join(checkpoint_path,"2",'test target.pt'))
    test_predict_2 = torch.load(os.path.join(checkpoint_path,"2",'test predict.pt'))

    test_target_3 = torch.load(os.path.join(checkpoint_path,"3",'test target.pt'))
    test_predict_3 = torch.load(os.path.join(checkpoint_path,"3",'test predict.pt'))

    test_target_4 = torch.load(os.path.join(checkpoint_path,"4",'test target.pt'))
    test_predict_4 = torch.load(os.path.join(checkpoint_path,"4",'test predict.pt'))

    test_target_5 = torch.load(os.path.join(checkpoint_path,"5",'test target.pt'))
    test_predict_5 = torch.load(os.path.join(checkpoint_path,"5",'test predict.pt'))

    checkpoint_path_sum = os.path.join(checkpoint_path,"sum")

    test_target_1 = torch.tensor(test_target_1) if not isinstance(test_target_1, torch.Tensor) else test_target_1
    test_target_2 = torch.tensor(test_target_2) if not isinstance(test_target_2, torch.Tensor) else test_target_2
    test_target_3 = torch.tensor(test_target_3) if not isinstance(test_target_3, torch.Tensor) else test_target_3
    test_target_4 = torch.tensor(test_target_4) if not isinstance(test_target_4, torch.Tensor) else test_target_4
    test_target_5 = torch.tensor(test_target_5) if not isinstance(test_target_5, torch.Tensor) else test_target_5
    test_predict_1 = torch.tensor(test_predict_1) if not isinstance(test_predict_1, torch.Tensor) else test_predict_1
    test_predict_2 = torch.tensor(test_predict_2) if not isinstance(test_predict_2, torch.Tensor) else test_predict_2
    test_predict_3 = torch.tensor(test_predict_3) if not isinstance(test_predict_3, torch.Tensor) else test_predict_3
    test_predict_4 = torch.tensor(test_predict_4) if not isinstance(test_predict_4, torch.Tensor) else test_predict_4
    test_predict_5 = torch.tensor(test_predict_5) if not isinstance(test_predict_5, torch.Tensor) else test_predict_5

    test_target = torch.cat([test_target_1, test_target_2, test_target_3, test_target_4, test_target_5], dim=0)
    test_predict = torch.cat([ test_predict_1, test_predict_2, test_predict_3, test_predict_4, test_predict_5], dim=0)

    assert len(test_target) == len(test_predict), f"预测结果长度 ({len(test_predict)}) 与目标长度 ({len(test_target)}) 不一致"

    Class_labels = ['standing', 'running', 'grazing', 'trotting', 'walking']


    def show_confusion_matrix(validations, predictions):
        # 先计算原始混淆矩阵（计数）
        cm = confusion_matrix(validations, predictions)

        # 计算每行的总数
        row_sums = cm.sum(axis=1)  # 每个真实标签的样本数

        # 为了防止某一行全为0导致除0错误，先复制一份并将0替换为1
        safe_row_sums = row_sums.astype(float).copy()
        zero_mask = safe_row_sums == 0
        safe_row_sums[zero_mask] = 1.0

        # 计算行内精度：cm[i,j] / row_sums[i]
        cm_precision = cm.astype(float) / safe_row_sums[:, None]
        # 如果某行本来没样本，就把这一行的精度全置为 0
        cm_precision[zero_mask, :] = 0.0

        # 乘以100 得到百分比
        cm_percent = cm_precision * 100

        # 把百分比做成 “17.8%” 这样的字符串
        annot_labels = [
            [f"{cm_percent[i, j]:.1f}%" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]

        # y 轴标签上显示类别索引＋样本数
        y_labels_with_counts = [
            f"{i}\n(n={row_sums[i]})" for i in range(len(row_sums))
        ]

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm_precision,               # 用 cm_precision 决定方块颜色
            cmap="coolwarm",
            linecolor="white",
            linewidths=1,
            xticklabels=[str(i) for i in range(cm.shape[1])],
            yticklabels=y_labels_with_counts,
            annot=annot_labels,          # 传入的是已经带“%”的字符串
            fmt="",                      # 改成空字符串，否则 Seaborn 会尝试把注释当数字来格式化
            vmin=0.0,
            vmax=1.0,
        )
        plt.title("Confusion Matrix (Row‐wise Precision)")
        plt.ylabel("True Label (Count)")
        plt.xlabel("Predicted Label")

        # 保存到磁盘
        cm_figuresavedpath = os.path.join(checkpoint_path_sum, "Confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_figuresavedpath), exist_ok=True)
        plt.tight_layout()
        plt.savefig(cm_figuresavedpath, dpi=600, bbox_inches='tight')
        pdf_path = cm_figuresavedpath.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=600, format='pdf', bbox_inches='tight')
        plt.close()

    # 总的保存路径
    checkpoint_path_sum = os.path.join(checkpoint_path, "sum")
    os.makedirs(checkpoint_path_sum, exist_ok=True)


    show_confusion_matrix(test_target, test_predict)

    accuracy_test = test_target.eq(test_predict).sum().item() / len(test_target)

    out_txtsavedpath = os.path.join(checkpoint_path_sum,'output.txt')
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(out_txtsavedpath), exist_ok=True)
    f = open(out_txtsavedpath, 'w+')
    print('Testing network......', file=f)
    print('Test set: Accuracy: {:.5f}, '.format(
            accuracy_test,
            ), file=f)
        
        #Obtain f1_score of the prediction
    fs_test = f1_score(test_target, test_predict, average='macro')
    print('f1 score = {:.5f}'.format(fs_test), file=f)
        
    kappa_value = cohen_kappa_score(test_target, test_predict)
    print("kappa value = {:.5f}".format(kappa_value), file=f)
        
    precision_test = precision_score(test_target, test_predict, average='macro')
    print('precision = {:.5f}'.format(precision_test), file=f)
        
    recall_test = recall_score(test_target, test_predict, average='macro')
    print('recall = {:.5f}'.format(recall_test), file=f)
        
    #Output the classification report
    print('------------', file=f)
    print('Classification Report', file=f)
    print(classification_report(test_target, test_predict), file=f)
    f.close()
        
   