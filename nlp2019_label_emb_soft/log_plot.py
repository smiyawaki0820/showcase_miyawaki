### loss_data_file を指定してグラフをプロット ###
import matplotlib
import matplotlib.pyplot as plt

def loss_plot(loss_file):
   
    matplotlib.use('Agg')
    train_iter = []
    loss_lis = []
    temp = float(0)
    count = 0
    with open(loss_file,'r') as f, open('loss.txt', 'w') as new_loss:
        for i,line in enumerate(f, start=1):
            #train_iter.append(i)
            if i % 23916 == 0:
                print(line)
                new_line = line.rstrip()
                loss = float(new_line)
                #loss = float(new_line) - temp
                #temp = float(new_line)
                loss_lis.append(loss)
                print(str(loss), file=new_loss, end='\n')
                count += 1
                train_iter.append(count)
    plt.plot(train_iter, loss_lis)
    plt.savefig('./work/loss_.png')

if __name__ == "__main__":
    loss_plot('result/log/model-e2e-stack_ve256_vu256_10_adam_lr0.0002_du0.1_dh0.0_True_size100_sub0_th0.6_it3_total_loss.txt')
