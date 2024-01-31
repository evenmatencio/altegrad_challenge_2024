ALTeGraD-2023 Data Challenge

Battre_Sacha_et_Wan

Syntax for the use of loss functions:
`loss = LossFunctions.Contrastive_Loss`

ou

`loss = LossFunctions.InfoNCE`

ou

`loss = LossFunctions.NTXent('cpu', batch_size, 0.1, True)`

Puis
`train_val_test.train(nb_epochs, optimizer, loss, model, train_loader, val_loader, save_path, device, hyper_param, print_every=1)`

To use the learning rate scheduler (LROnPlateau):

`optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)`
`scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, threshold=0.1, threshold_mode='rel', verbose=True)`

and when calling the training:
`train(nb_epochs, optimizer, loss, model, train_loader, val_loader, save_path, device, hyper_param, save_id=1000, scheduler=scheduler, print_every=1)`
(the default value of the scheduler is None, so that it works even when not given)
