from torch import save

def save_checkpoint(model, optimizer, amp, scheduler, dev_eval_dict, model_path):
    print('Saving model to: %s' % model_path)
    if amp is not None:
        print('amp is not None')
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'amp_state_dict': amp.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict(),
                 'dev_eval_dict': dev_eval_dict}
    else:
        print('amp is None')
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'amp_state_dict': amp,
                 'scheduler_state_dict': scheduler.state_dict(),
                 'dev_eval_dict': dev_eval_dict}
    save(state, model_path)