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
