import torch
import logging
from tqdm import tqdm

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')

def evaluate_model(model, eval_loader, device):
    model.eval()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    total_mape_loss = 0.0

    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(eval_loader,desc="Evaluating model", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse_loss = mse_loss_fn(outputs, targets)
            total_mse_loss += mse_loss.item() * inputs.size(0)

            mae_loss = mae_loss_fn(outputs, targets)
            total_mae_loss += mae_loss.item() * inputs.size(0)

            total_samples += inputs.size(0)

    avg_eval_mse_loss = total_mse_loss / total_samples
    avg_eval_mae_loss = total_mae_loss / total_samples

    logging.info(f'EVAL:::MSE: {avg_eval_mse_loss:.7f}, MAE: {avg_eval_mae_loss:.7f}')
    return avg_eval_mse_loss, avg_eval_mae_loss

def test_model(model, test_loader, device):
    model.eval()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    total_mse_loss = 0.0
    total_mae_loss = 0.0

    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader,desc="Testing model", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse_loss = mse_loss_fn(outputs, targets)
            total_mse_loss += mse_loss.item() * inputs.size(0)

            mae_loss = mae_loss_fn(outputs, targets)
            total_mae_loss += mae_loss.item() * inputs.size(0)

            total_samples += inputs.size(0)
    avg_test_mse_loss = total_mse_loss / total_samples
    avg_test_mae_loss = total_mae_loss / total_samples
    logging.info(f'TEST:::MSE: {avg_test_mse_loss:.7f}, MAE: {avg_test_mae_loss:.7f}')
    return avg_test_mse_loss, avg_test_mae_loss


def train_model(model, train_loader, eval_loader, criterion, optimizer, device, args):
    logs_path = args.log_path
    setup_logging(logs_path)

    if args.if_pretrained == True:
        model.load_state_dict(torch.load(args.checkpoints))
        print(f'Loaded model from {args.checkpoints}')

    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        count = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            progress_bar.set_postfix(loss=loss.item())

            count += 1
            if count % args.log_per_samples == 0:
                logging.info(f'TRAIN:::sample {count + 1} - Train Loss: {loss.item():.4f}')

        average_loss = total_loss / total_samples
        logging.info(f'TRAIN FINISHED:::Epoch {epoch + 1}, Train MSE Loss: {average_loss:.7f}')
        print(f'TRAIN FINISHED:::Epoch {epoch + 1}, Train MSE Loss: {average_loss:.7f}')
        eval_mse_loss, eval_mae_loss = evaluate_model(model, eval_loader, device)
        print(f'EVAL:::MSE Loss: {eval_mse_loss:.7f}, MAE Loss: {eval_mae_loss:.7f}')

        if eval_mse_loss < best_loss:
            best_loss = eval_mse_loss
            logging.info(f'New best model found at epoch {epoch + 1} with Eval Loss: {best_loss:.7f}.')
            print(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), args.checkpoints)

    print("Training complete.")
