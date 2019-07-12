def find_proper_lr(learner):
    """VERIFIED by kaggle, result improved
    """
    learner.lr_find()
    learner.recorder.plot(suggestion=True)
    
    
def show_all(learner, print=print):
    """VERIFIED by kaggle, result dropped, why? may be deviation
    显示learner在训练后的所有情况
    """
    if len(learner.metrics) > 0:
        learner.recorder.plot_metrics()
        print('<---Metrics--->')
        print(learner.recorder.metrics)
    else:
        print('<---NO Metrics--->')
    
    learner.recorder.plot_losses()
#     print('<---Train Losses--->')
#     print(learner.recorder.losses)
    print('<---Val Losses--->')
    print(learner.recorder.val_losses)
    
    learner.recorder.plot_lr(show_moms=True)
