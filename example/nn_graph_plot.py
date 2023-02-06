import graphviz as gz
import random


def neural_graph(inp=3, hide=(10,), outp=3, inp_label='input', hide_label='hide',
                 outp_label='output', dropout=True,
                 style='v', size='2, 1'):
    """ 绘制简易神经网络图（有向图）
    :param inp: 输入神经元个数
    :param hide: 隐藏层神经元个数, 可迭代数组
    :param outp: 输出神经元个数
    :param inp_label: 输入名称显示
    :param hide_label: 隐藏层名称显示
    :param outp_label: 输出名称显示
    :param dropout: 是否全连接
    :param style: 水平或垂直显示， 可选项为 'h', 'v'
    :param size: 图像显示大小
    :return: 有向图 """
    dot = gz.Digraph(name='neural network')
    dot.attr(size=size)
    if style == 'v': dot.attr(rankdir='LR')

    def draw(enter, exit, label1, label2):
        for i in range(enter):
            for j in range(exit):
                if dropout:
                    if random.randint(0, max(enter, exit)):
                        dot.edge('%s%s' % (label1, i), '%s%s' % (label2, j))
                    else:
                        dot.edge('%s%s' % (label1, i), '%s%s' % (label2, j))

    hide = list(hide)
    hide.insert(0, inp)
    hide.append(outp)
    for index, (i, j) in enumerate(zip(hide[:-1], hide[1:])):
        if index == 0:
            draw(i, j, inp_label, hide_label + str(index))
        elif index == len(hide) - 2:
            draw(i, j, hide_label + str(index - 1), outp_label)
        else:
            draw(i, j, hide_label + str(index - 1), hide_label + str(index))
    return dot


if __name__ == '__main__':
    nn_dot = neural_graph()
    nn_dot.view()
