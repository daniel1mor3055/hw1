import torch


def replace_final_layer(model, config, model_name, device):
    hidden_dim = config["models"][model_name]["hidden_dim"]
    embed_dim = config["models"][model_name]["embed_dim"]
    if model_name == "lstm":
        output_dim = 1
        model.Wy = torch.nn.Parameter(torch.empty(output_dim, hidden_dim).to(device))
        model.by = torch.nn.Parameter(torch.zeros(output_dim, 1).to(device))
        torch.nn.init.xavier_uniform_(model.Wy)
    elif model_name == "s4":
        output_dim = 1
        model.decoder = torch.nn.Linear(hidden_dim, output_dim).to(device)
    else:
        output_dim = 1
        model.fc_out = torch.nn.Linear(embed_dim, output_dim).to(device)
