### loss_data_file を指定してグラフをプロット ###
import matplotlib
import matplotlib.pyplot as plt

def loss_plot(loss_file):
   
    matplotlib.use('Agg')
    train_iter = []
    loss_lis = []
    with open(loss_file,'r') as f:
        for i,line in enumerate(f):
            train_iter.append(i)
            new_line = line.rstrip()
            loss = float(new_line)
            loss_lis.append(loss)

    plt.xlabel('train_iter')
    plt.ylabel('loss')
    plt.plot(train_iter, loss_lis)
    plt.savefig('./work/loss.png')

if __name__ == "__main__":
    loss_plot('./work/loss.txt')
