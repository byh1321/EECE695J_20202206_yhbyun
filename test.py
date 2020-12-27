import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from src.utils import count_acc, Averager, csv_write, square_euclidean_metric, progress_bar
from model import FewShotModel, ProtoNet, ResNet18, AlexNet
import torch.backends.cudnn as cudnn
import pdb
import wideresnet

import time
import torch.nn.functional as F
from src.test_dataset import CUB as Test_Dataset
from src.test_sampler import Test_Sampler

" User input value "
TOTAL = 10000  # total step of training
PRINT_FREQ = 5  # frequency of print loss and accuracy at training step
VAL_FREQ = 100  # frequency of model eval on validation dataset
SAVE_FREQ = 100  # frequency of saving model
TEST_SIZE = 200  # fixed

" fixed value "
VAL_TOTAL = 100

def Test_phase(args, k):

    model1 = AlexNet()
    model1.cuda()
    model2 = AlexNet()
    model2.cuda()
    model3 = AlexNet()
    model3.cuda()
    model4 = AlexNet()
    model4.cuda()
    state_dict = torch.load('checkpoints/alexnet_63.pth')
    model1.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6240.pth')
    model2.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6215.pth')
    model3.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6085.pth')
    model4.load_state_dict(state_dict)

    model1 = torch.nn.DataParallel(model1, device_ids=range(torch.cuda.device_count()))
    model2 = torch.nn.DataParallel(model2, device_ids=range(torch.cuda.device_count()))
    model3 = torch.nn.DataParallel(model3, device_ids=range(torch.cuda.device_count()))
    model4 = torch.nn.DataParallel(model4, device_ids=range(torch.cuda.device_count()))

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    csv = csv_write(args)

    dataset = Test_Dataset(args.dpath)
    test_sampler = Test_Sampler(dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    test_loader = DataLoader(dataset=dataset, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    print('Test start!')
    for i in range(TEST_SIZE):
        for episode in test_loader:
            data = episode.cuda()

            data_shot, data_query = data[:k], data[k:]

            """ TEST Method """
            """ Predict the query images belong to which classes
            
            At the training phase, you measured logits. 
            The logits can be distance or similarity between query images and 5 images of each classes.
            From logits, you can pick a one class that have most low distance or high similarity.
            
            ex) # when logits is distance
                pred = torch.argmin(logits, dim=1)
            
                # when logits is prob
                pred = torch.argmax(logits, dim=1)
                
            pred is torch.tensor with size [20] and the each component value is zero to four
            """

            model1.eval()
            features_shot = model1(data_shot)
            n_sample = int(args.query/args.nway)
            features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
            for j in range(int(args.nway)):
                start = j*args.kshot
                end = (j+1)*args.kshot
                features_shot_mean[j] = features_shot[start:end].mean(dim=0)

            features_query = model1(data_query)
            logits1 = square_euclidean_metric(features_query, features_shot_mean)

            model2.eval()
            features_shot = model2(data_shot)
            n_sample = int(args.query/args.nway)
            features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
            for j in range(int(args.nway)):
                start = j*args.kshot
                end = (j+1)*args.kshot
                features_shot_mean[j] = features_shot[start:end].mean(dim=0)

            features_query = model2(data_query)
            logits2 = square_euclidean_metric(features_query, features_shot_mean)

            model3.eval()
            features_shot = model3(data_shot)
            n_sample = int(args.query/args.nway)
            features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
            for j in range(int(args.nway)):
                start = j*args.kshot
                end = (j+1)*args.kshot
                features_shot_mean[j] = features_shot[start:end].mean(dim=0)

            features_query = model3(data_query)
            logits3 = square_euclidean_metric(features_query, features_shot_mean)
            
            model4.eval()
            features_shot = model4(data_shot)
            n_sample = int(args.query/args.nway)
            features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
            for j in range(int(args.nway)):
                start = j*args.kshot
                end = (j+1)*args.kshot
                features_shot_mean[j] = features_shot[start:end].mean(dim=0)

            features_query = model4(data_query)
            logits4 = square_euclidean_metric(features_query, features_shot_mean)

            logits = (logits1 + logits2 + logits3 + logits4)/4

            lsoft = F.log_softmax(-logits, dim=1).view(args.kshot, n_sample, -1)
            pred = torch.argmin(logits,dim=1)

            # save your prediction as StudentID_Name.csv file
            csv.add(pred)

    csv.close()
    print('Test finished, check the csv file!')
    exit()

def val(args):
    # the number of N way, K shot images
    k = args.nway * args.kshot

    """ TODO 1.a """
    " Make your own model for Few-shot Classification in 'model.py' file."

    """ TODO 1.a END """

    model1 = AlexNet()
    model1.cuda()
    model2 = AlexNet()
    model2.cuda()
    model3 = AlexNet()
    model3.cuda()
    model4 = AlexNet()
    model4.cuda()
    state_dict = torch.load('checkpoints/alexnet_63.pth')
    model1.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6240.pth')
    model2.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6215.pth')
    model3.load_state_dict(state_dict)
    state_dict = torch.load('checkpoints/alexnet_6085.pth')
    model4.load_state_dict(state_dict)
    #state_dict = torch.load('')
    #model.load_state_dict(state_dict)
    #model1 = model.clone()

    model1 = torch.nn.DataParallel(model1, device_ids=range(torch.cuda.device_count()))
    model2 = torch.nn.DataParallel(model2, device_ids=range(torch.cuda.device_count()))
    model3 = torch.nn.DataParallel(model3, device_ids=range(torch.cuda.device_count()))
    model4 = torch.nn.DataParallel(model4, device_ids=range(torch.cuda.device_count()))

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    csv = csv_write(args)

    # Validation data loading
    val_dataset = Dataset(args.dpath, state='val')
    val_sampler = Sampler(val_dataset._labels, n_way=args.nway, k_shot=args.kshot, query=args.query)
    val_data_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    """ TODO 1.b (optional) """
    " Set an optimizer or scheduler for Few-shot classification (optional) "

    # Default optimizer setting
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    """ TODO 1.b (optional) END """

    tl = Averager()  # save average loss
    ta = Averager()  # save average accuracy

    print('test start')
    test_correct = 0
    test_total = 0
    test_loss = 0

    # validation start
    print('validation start')
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    with torch.no_grad():
        vl = Averager()  # save average loss
        va = Averager()  # save average accuracy
        for j in range(VAL_TOTAL):
            for episode in val_data_loader:
                data, label = [_.cuda() for _ in episode]

                data_shot, data_query = data[:k], data[k:] # load an episode

                label_shot, label_query = label[:k], label[k:]
                label_shot = sorted(list(set(label_shot.tolist())))

                label_query = label_query.tolist()

                labels = []

                for j in range(len(label_query)):
                    label = label_shot.index(label_query[j])
                    labels.append(label)
                labels = torch.tensor(labels).cuda()

                """ TODO 2 ( Same as above TODO 2 ) """
                """ Train the model 
                Input:
                    data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                                be careful when using torch.reshape or .view functions
                    data_query : torch.tensor, query images, [args.query, 3, h, w]
                    labels : torch.tensor, labels of query images, [args.query]
                output:
                    loss : torch scalar tensor which used for updating your model
                    logits : A value to measure accuracy and loss
                """

                data, label = [_.cuda() for _ in episode]  # load an episode

                # split an episode images and labels into shots and query set
                # note! data_shot shape is ( nway * kshot, 3, h, w ) not ( kshot * nway, 3, h, w )
                # Take care when reshape the data shot
                data_shot, data_query = data[:k], data[k:]

                #print('label : ',label)

                label_shot, label_query = label[:k], label[k:]
                label_shot = sorted(list(set(label_shot.tolist())))

                # convert labels into 0-4 values
                label_query = label_query.tolist()

                labels = []
                for j in range(len(label_query)):
                    label = label_shot.index(label_query[j])
                    labels.append(label)
                labels = torch.tensor(labels).cuda()

                """ TODO 2 ( Same as above TODO 2 ) """
                """ Make a loss function and train your own model
                Input:
                    data_shot : torch.tensor, shot images, [args.nway * args.kshot, 3, h, w]
                                be careful when using torch.reshape or .view functions
                    (25, 3, 400, 400)
                    data_query : torch.tensor, query images, [args.query, 3, h, w]
                    (20, 3, 400, 400)
                    labels : torch.tensor, labels of query images, [args.query]
                    (20)
                output:
                    loss : torch scalar tensor which used for updating your model
                    logits : A value to measure accuracy and loss
                """

                model1.eval()
                features_shot = model1(data_shot)
                n_sample = int(args.query/args.nway)
                features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
                for j in range(int(args.nway)):
                    start = j*args.kshot
                    end = (j+1)*args.kshot
                    features_shot_mean[j] = features_shot[start:end].mean(dim=0)

                features_query = model1(data_query)
                logits1 = square_euclidean_metric(features_query, features_shot_mean)

                model2.eval()
                features_shot = model2(data_shot)
                n_sample = int(args.query/args.nway)
                features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
                for j in range(int(args.nway)):
                    start = j*args.kshot
                    end = (j+1)*args.kshot
                    features_shot_mean[j] = features_shot[start:end].mean(dim=0)

                features_query = model2(data_query)
                logits2 = square_euclidean_metric(features_query, features_shot_mean)

                model3.eval()
                features_shot = model3(data_shot)
                n_sample = int(args.query/args.nway)
                features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
                for j in range(int(args.nway)):
                    start = j*args.kshot
                    end = (j+1)*args.kshot
                    features_shot_mean[j] = features_shot[start:end].mean(dim=0)

                features_query = model3(data_query)
                logits3 = square_euclidean_metric(features_query, features_shot_mean)
                
                model4.eval()
                features_shot = model4(data_shot)
                n_sample = int(args.query/args.nway)
                features_shot_mean = torch.zeros(args.nway, features_shot.size(1)).cuda()
                for j in range(int(args.nway)):
                    start = j*args.kshot
                    end = (j+1)*args.kshot
                    features_shot_mean[j] = features_shot[start:end].mean(dim=0)

                features_query = model4(data_query)
                logits4 = square_euclidean_metric(features_query, features_shot_mean)

                logits = (logits1+logits2+logits3+logits4)/4

                lsoft = F.log_softmax(-logits, dim=1).view(args.kshot, n_sample, -1)
                pred = torch.argmin(logits,dim=1)
                
                """ TODO 2 END """

                acc = count_acc(logits, labels)

                va.add(acc)
                csv.add(pred)

                proto = None; logits = None; loss = None

        print('val accuracy mean : %.4f' % va.item())
        print('val loss mean : %.4f' % vl.item())

        # initialize loss and accuracy mean
        vl = None
        va = None
        vl = Averager()
        va = Averager()

    csv.close()
    print('Test finished, check the csv file!')
    exit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='model', help="name your experiment")
    parser.add_argument('--lr', default=0.001, type=float, help="set the learning rate parameter")
    parser.add_argument('--se', default=100, type=int,help="starting count")
    parser.add_argument('--dpath', '--d', default='/home2/CUB_200_2011/CUB_200_2011', type=str,
                        help='the path where dataset is located')
    parser.add_argument('--restore_ckpt', type=str, help="checkpoint/100_model.pth")
    parser.add_argument('--nway', '--n', default=5, type=int, help='number of class in the support set (5 or 20)')
    parser.add_argument('--kshot', '--k', default=5, type=int,
                        help='number of data in each class in the support set (1 or 5)')
    parser.add_argument('--query', '--q', default=20, type=int, help='number of query data')
    parser.add_argument('--ntest', default=100, type=int, help='number of tests')
    parser.add_argument('--gpus', type=int, nargs='+', default=1)
    parser.add_argument('--test_mode', type=int, default=0, help="if you want to test the model, change the value to 1")

    args = parser.parse_args()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    #torch.cuda.set_device(7)

    #val(args)
    Test_phase(args, 25)

