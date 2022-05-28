import pandas as pd
import plotly.express as px

def plot_model_fit_loss(train_loss, val_loss, vertical_line=None, subtitle=""):
    """
    Given the output of the fit model it plots an interactive dataviz of the results (loss)
    """

    # create dataframe suitable for plotly
    df_train_acc = pd.DataFrame(
        list(zip(list(range(1,len(train_loss)+1)),train_loss)),
        columns =["epoch","loss"]
        )
    df_train_acc["type"] = "train"

    df_val_acc = pd.DataFrame(
        list(zip(list(range(1,len(val_loss)+1)),val_loss)),
        columns =["epoch","loss"]
        )
    df_val_acc["type"] = "val"

    res_df = pd.concat([df_train_acc,df_val_acc]).reset_index(drop=True)

    # create representation
    fig = px.line(
        res_df, 
        x="epoch", 
        y="loss", 
        color='type', 
        markers=True,
        title="Model Loss<br><sup>" + subtitle +"<sup>",
        width=1000, height=600)

    if vertical_line is not None:
        fig.add_vline(x=vertical_line, line_width=2, line_dash="dash", line_color="green")
        
    fig.update_xaxes(range=[0.75, max(res_df["epoch"])+0.25])
    fig.update_yaxes(range=(-0.025, 1.65),constrain='domain')
    fig.show()


def plot_classes_accuracy(classes_res_df, vertical_line=None, subtitle=""):
    """
    Given the output of the fit model it plots an interactive dataviz of the results (loss)
    """

    # obtain lists for each class
    classes_res_df = classes_res_df.reset_index()

    # class 0
    class_0 = list(classes_res_df.loc[classes_res_df['index'] == 0]['accuracy'])

    # class 1
    class_1 = list(classes_res_df.loc[classes_res_df['index'] == 1]['accuracy'])

    # class 2
    class_2 = list(classes_res_df.loc[classes_res_df['index'] == 2]['accuracy'])

    # create dataframe suitable for plotly
    df_class_0 = pd.DataFrame(
        list(zip(list(range(1,len(class_0)+1)),class_0)),
        columns =["epoch","accuracy"]
        )
    df_class_0["Class"] = "0"

    df_class_1 = pd.DataFrame(
        list(zip(list(range(1,len(class_1)+1)),class_1)),
        columns =["epoch","accuracy"]
        )
    df_class_1["Class"] = "1"

    df_class_2 = pd.DataFrame(
        list(zip(list(range(1,len(class_2)+1)),class_2)),
        columns =["epoch","accuracy"]
        )
    df_class_2["Class"] = "2"

    res_df = pd.concat([df_class_0,df_class_1,df_class_2]).reset_index(drop=True)

    # create representation
    fig = px.line(
        res_df, 
        x="epoch", 
        y="accuracy", 
        color='Class', 
        markers=True,
        title="Classes Accuracy<br><sup>" + subtitle +"<sup>",
        width=1000, height=600)

    if vertical_line is not None:
        fig.add_vline(x=vertical_line, line_width=2, line_dash="dash", line_color="green")

    fig.update_xaxes(range=[0.75, max(res_df["epoch"])+0.25])
    fig.update_yaxes(range=(-0.015, 1.015),constrain='domain')
    fig.show()