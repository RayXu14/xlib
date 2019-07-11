def find_proper_lr(learner)
    learner.lr_find()
    learner.recorder.plot(suggestion=True)

def show_all(learner, print=print):
    """
    显示learner在训练后的所有情况
    """
    learner.plot_metrics()
    print('<---Metrics--->')
    print(learner.recorder.metrics)
    
    learner.recorder.plot_losses()
    print('<---Train Losses--->')
    print(learner.recorder.train_losses)
    print('<---Val Losses--->')
    print(learner.recorder.val_losses)
    
    learner.recorder.plot_lr(show_moms=True)
