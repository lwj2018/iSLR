from tensorboard.backend.event_processing import event_accumulator
 
#加载日志数据
ea=event_accumulator.EventAccumulator(
    '../runs/BNInception_class100/events.out.tfevents.1571665597.ITSK-20190902DK') 
ea.Reload()
 
train_loss=ea.scalars.Items('train/loss')
train_acc=ea.scalars.Items('train/acc')
print([(i.step,i.value) for i in train_loss if i<5])