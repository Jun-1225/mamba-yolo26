from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/root/mamba-yolo26/ultralytics/cfg/datasets/dacl10k.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str, default='/root/mamba-yolo26/ultralytics/cfg/models/26/yolo26m-seg.yaml', help='model path(s)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', default='auto', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', default=True, help='open amp')
    parser.add_argument('--project', default='/root/mamba-yolo26/out_dir/yolo26', help='save to project/name')
    parser.add_argument('--name', default='yolo-mamba', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate (i.e. SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--cos_lr', type=bool, default=True, help='use cosine lr schedule')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='use warmup_epochs')
    parser.add_argument('--close_mosaic', type=int, default=5, help='use close_mosaic')
    


    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": opt.name,
        'imgsz':opt.imgsz,
        'lr0':opt.lr0,
        'cos_lr':opt.cos_lr,
        'close_mosaic' :opt.close_mosaic,
        'warmup_epochs':opt.warmup_epochs
    }
    model_conf = opt.config
    if task == "train":
        # 训练时：加载 yaml 配置文件（从零开始），或者已有的 pt 权重（继续训练）
        model = YOLO(opt.config)
        model.train(**args)

        
    elif task == "val":
        # 验证时：必须加载训练好的 .pt 权重！
        if not opt.config.endswith('.pt'):
            print("⚠️ 警告: 验证模式下，--config 应该指向你训练好的 best.pt 权重文件，而不是 yaml 文件！")
            exit() 
        model = YOLO(opt.config)
        # 验证时通常不需要传 epochs, optimizer 等训练参数
        val_args = {"data": opt.data, "batch": opt.batch_size, "imgsz": opt.imgsz, "device": opt.device}
        model.val(**val_args)
