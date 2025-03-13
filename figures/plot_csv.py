import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def plot_line(x, y, legends, xtick_labels, title, xlabel, ylabel, loc, savefile=None):
    # 绘图设置
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    line_color = ['red', 'blue', '#19FF19', '#FF03FF', 'green']
    line_marker = ['o', 'x', 'v', 's', '*']

    # 设置图像的x轴刻度与刻度标签
    x_ticks = list(np.arange(len(x)))  # 当x轴标签与x轴刻度相同时，x=x_ticks
    plt.xticks(x, labels=xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)

    plt.plot(x, y[0], label=legends[0],
             color=line_color[0], linewidth=1.5, linestyle='-',  # 线条颜色类型、线宽、线型
             marker=line_marker[0], markersize=8, markevery=1  # 标记类型、大小
             )

    for i in range(1, len(legends)):
        plt.plot(x, y[i], label=legends[i],
                 color=line_color[i], linewidth=1.5, linestyle='--',  # 线条颜色类型、线宽、线型
                 marker=line_marker[i], markersize=8, markevery=1  # 标记类型、大小
                 )

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # plt.legend(bbox_to_anchor=(1.0, 0.1), loc=loc, fontsize=14)
    plt.legend(loc=loc, fontsize=14)
    plt.grid()
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile, dpi=300)


def plot_hatchbar(x, y, legends, xtick_labels, title, xlabel, ylabel, loc, savefile=None):
    # 绘图设置
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    line_color = ['green', 'blue', 'gray', 'red']
    color = ['white', 'white', 'white', 'white']
    bar_hacth = ['xx', '..', '\\\\', 'oo']
    bar_width = 0.15

    # 设置位置
    xtick_loc = list(np.arange(len(x)))

    # 绘制条形图
    bars = []

    for i in range(len(legends)):
        xtick_loc = [x + bar_width + 0.05 for x in xtick_loc]
        bar = plt.bar(xtick_loc, y[i], label=legends[i],
                      width=bar_width, hatch=bar_hacth[i],  # 条形宽度、填充纹理
                      edgecolor=line_color[i], color=color[i]  # 线条颜色、填充颜色
        )
        bars += bar

    # 设置刻度标签
    plt.ylim(0.84, 1.01)
    plt.xticks([r + bar_width * 2 + 0.2 for r in range(len(x))], xtick_labels, fontsize=14)
    plt.yticks(fontsize=14)

    # # 添加数值标签
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2 + 0.1, height, f'{height: }', ha='center', va='bottom')

    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(loc=loc, fontsize=14, ncols=2)
    plt.grid(axis='y', linewidth=0.5, linestyle='--')
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile, dpi=300)


def plot_2ybar(x, y1, y2, legends, title, xlabel, loc, savefile=None):
    # 绘图设置
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    line_color = ['green', 'blue', 'gray', 'red']
    color = ['white', 'white', 'white', 'white']
    bar_hacth = ['xx', '..', '\\\\', 'oo']
    bar_width = 0.35
    # # 设置全局 hatch 线条宽度为 2
    # mpl.rcParams['hatch.linewidth'] = 2
    # 设置位置
    xtick_loc = list(np.arange(len(x)))

    # 创建一个图形和一个坐标轴对象
    fig, ax1 = plt.subplots()

    # 绘制第一个 y 轴的条形图
    ax1.bar(xtick_loc, y1, label=legends[0],
                      width=bar_width, hatch=bar_hacth[0],  # 条形宽度、填充纹理
                      edgecolor=line_color[0], color=color[0])  # 线条颜色、填充颜色

    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    # 绘制第二个 y 轴的条形图
    xtick_loc = [x + bar_width + 0.05 for x in xtick_loc]
    ax2.bar(xtick_loc, y2, label=legends[1],
                      width=bar_width, hatch=bar_hacth[1],  # 条形宽度、填充纹理
                      edgecolor=line_color[1], color=color[1])  # 线条颜色、填充颜色

    # 设置 y 轴的颜色
    ax1.tick_params('y', labelsize=14, colors=line_color[0])
    ax1.spines['left'].set_color(line_color[0])
    ax2.spines['left'].set_visible(False)
    ax2.tick_params('y', labelsize=14, colors=line_color[1])
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_color(line_color[1])
    # 删除上框线
    # ax1.spines['top'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)

    # 设置x轴刻度字体大小
    plt.xticks([r + bar_width - 0.15 for r in range(len(x))], x)
    ax1.tick_params('x', labelsize=14)
    ax2.tick_params('x', labelsize=14)

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=loc, fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=16)
    plt.title(title, fontsize=18)
    # plt.grid(axis='y', linewidth=0.5, linestyle='--')
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile, dpi=300)


if __name__ == '__main__':

    # --------------------------------- Feature Selection 折线图 ---------------------------------
    y = np.array([[0.9505,0.9757,0.9775,0.9806],
                  [0.9181,0.9767,0.9741,0.9801],
                  [0.9641,0.9741,0.9806,0.9801],
                  [0.973,0.9733,0.972,0.9806],
                  [0.9455,0.9764,0.9738,0.9775]
                  ])

    # for row in range(y.shape[0]):
    #     mean_value = np.mean(y[row, :])
    #     std_value = np.std(y[row, :])
    #     y[row, :] = (y[row, :] - mean_value) / std_value
    # print(y)
    legends = ['SHAP-based', 'Univariate FS', 'RFE', 'RFI', 'IG']
    x = [1, 2, 3, 4]
    plot_line(x, y, legends,
              xtick_labels=['5', '10', '15', '20'],
              title='CICIDS2017 Dataset',
              xlabel='Number of Node Features',
              ylabel='Accuracy',
              loc='best',
              savefile='FS_CICIDS2017.png'
              )

    # --------------------------------- Performance 条形图 ---------------------------------
    # BoT-IoT
    # y = [[0.9465, 0.9443, 0.9465, 0.9449],
    #      [0.9491, 0.9479, 0.9491, 0.9475],
    #      [0.9523, 0.9494, 0.9523, 0.9503],
    #      [0.9801, 0.979, 0.9801, 0.9789]]
    # CICIDS
    # y = [[0.967, 0.9568, 0.967, 0.9617],
    #      [0.9741, 0.9658, 0.9741, 0.9693],
    #      [0.9641, 0.9559, 0.9641, 0.9594],
    #      [0.9859, 0.9841, 0.9859, 0.9849]]
    # x = [1, 2, 3, 4]
    # plot_hatchbar(x, y,
    #               legends=['GAT', 'GCN', 'GCN+GAT', 'FIR-GNN'],
    #               xtick_labels=['AC', 'PR', 'RC', 'F1'],
    #               title='CICIDS2017 Dataset',
    #               xlabel='Metrics',
    #               ylabel='Values',
    #               loc='upper center',
    #               savefile='CICIDS.png')

    # # --------------------------------- 参数敏感性 条形图 ---------------------------------
    # # Time Window (s)
    # y1 = [6793.72,7377.01,7772.66,8277.63,8590.09]
    # y2 = [1555.04,2334.43,2927.18,3619.12,4227.79]
    # x = [10,20,30,40,50]
    #
    # plot_2ybar(x, y1, y2,
    #            legends=['BoT-IoT', 'CICIDS'],
    #            title='Max GPU Used (MB)',
    #            xlabel='Time Window (s)',
    #            loc='upper left',
    #            savefile='T_Max GPU Used.png')
