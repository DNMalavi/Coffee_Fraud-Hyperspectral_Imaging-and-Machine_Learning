# Coffee Adulteration Prediction App
# -------------------------------------------------------------
# This Shiny app uses near-infrared hyperspectral imaging (HSI) data
# (900–1700 nm) from instant Arabica coffee (medium roast) to predict
# the level of Robusta adulteration. It allows users to select between
# LASSO and PLS regression models, upload their data, and obtain
# adulteration predictions. The app also features variable importance
# plots, coefficients and wavelengths, downloadable example data,
# spectra visualisation, and model comparison outputs.
#
# The app is intended as a research prototype for NIR-HSI based
# quantification and is not yet validated for routine quality control.

# Packages
library(shiny)
library(caret)
library(glmnet)
library(readr)
library(readxl)
library(DT)
library(shinycssloaders)
library(shinythemes)
library(pls)
library(ggplot2)
library(tools)
library(dplyr)
library(tidyr)
library(viridis)

# Load models and column names
lasso_model    <- readRDS("fit_lasso_raw.rds")
pls_model      <- readRDS("fit_pls_raw.rds")
X_column_names <- readRDS("X_column_names.rds")  # e.g. "X935", "X940", ...

# UI
ui <- fluidPage(
  theme = shinytheme("cerulean"),
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f5f7f8;
      }
      .app-header {
        background: linear-gradient(to right, #111827, #1f2937, #6f42c1, #e83e8c);
        padding: 14px 20px 4px 20px;
        color: white;
        font-weight: 700;
        font-size: 24px;
        border-radius: 0 0 18px 18px;
        margin-bottom: 10px;
      }
      .app-header-inner {
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .app-title-main {
        display: flex;
        flex-direction: column;
      }
      .app-title-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .app-subtitle {
        font-size: 13px;
        font-weight: 400;
        opacity: 0.95;
        margin-top: 4px;
      }
      .app-ribbon {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border: 1px solid rgba(148, 163, 184, 0.7);
      }
      .hsi-strip {
        margin-top: 8px;
        width: 100%;
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(
          to right,
          #00004f,
          #001f7f,
          #0044ff,
          #00b3ff,
          #00ff8c,
          #a0ff00,
          #ffdd00,
          #ff8800,
          #ff0044
        );
        opacity: 0.9;
      }
      .main-panel-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,0.08);
        margin-bottom: 15px;
      }
    "))
  ),
  
  # Custom header
  div(
    class = "app-header",
    div(
      class = "app-header-inner",
      div(
        class = "app-title-main",
        div(
          class = "app-title-row",
          span("☕ SpectraQuant: Instant Coffee Adulteration Predictor")
        ),
        span(
          class = "app-subtitle",
          HTML("🌈 Near-Infrared Hyperspectral Imaging (900–1700 nm) · Chemometrics · Machine Learning Regression")
        ),
        div(class = "hsi-strip")
      ),
      div(
        class = "app-ribbon",
        "Research prototype · Not for routine QC"
      )
    )
  ),
  
  sidebarLayout(
    sidebarPanel(
      fileInput(
        "file",
        "Upload spectral data file",
        accept = c(".csv", ".txt", ".tsv", ".xls", ".xlsx")
      ),
      selectInput(
        "model_type",
        "Select regression model:",
        choices = c("LASSO", "PLS")
      ),
      textInput("filename", "Custom filename", "adulteration_predictions"),
      actionButton("predict", "🔍 Predict adulteration", class = "btn btn-primary"),
      actionButton("reset",   "🔄 Reset app",            class = "btn btn-warning"),
      hr(),
      downloadButton("downloadTemplate", "Download example file"),
      downloadButton("downloadData",     "Download predictions")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel(
          "📘 User guide",
          br(),
          div(
            class = "main-panel-card",
            h3("How to use SpectraQuant"),
            tags$ol(
              tags$li("Prepare your spectral data so each row is a sample and columns are NIR-HSI variables (e.g. X935–X1720)."),
              tags$li("Ensure the first column contains sample IDs (e.g. Sample_ID)."),
              tags$li("Upload the file using the \"Upload spectral data file\" control in the sidebar."),
              tags$li("Select the regression model (LASSO or PLS) and click \"Predict adulteration\"."),
              tags$li("Inspect the predictions, distribution plot, variable importance, and model cross-validation curve."),
              tags$li("Download predictions or the example file for demonstration and teaching.")
            ),
            p("Predictions represent the estimated percentage of Robusta adulteration in instant Arabica coffee samples."),
            p("This application represents a research-stage prototype for NIR-HSI quantification.")
          )
        ),
        
        tabPanel(
          "📁 Overview",
          br(),
          div(
            class = "main-panel-card",
            h4("File metadata & diagnostics"),
            verbatimTextOutput("file_info")
          ),
          div(
            class = "main-panel-card",
            h4("Preview of uploaded data (first 5 rows)"),
            withSpinner(DTOutput("data_preview"))
          ),
          div(
            class = "main-panel-card",
            h4("Predicted adulteration (%)"),
            withSpinner(DTOutput("predictions"))
          ),
          div(
            class = "main-panel-card",
            h4("Adulteration level distribution"),
            withSpinner(plotOutput("prediction_plot", height = "360px"))
          )
        ),
        
        tabPanel(
          "📡 Spectra viewer",
          br(),
          div(
            class = "main-panel-card",
            h4("Spectral profiles (raw NIR-HSI)"),
            tags$small("Visualise raw near-infrared spectra (935–1720 nm) for a subset of uploaded samples."),
            withSpinner(plotOutput("spectra_plot", height = "360px"))
          )
        ),
        
        tabPanel(
          "📉 Model cross-validation",
          br(),
          div(
            class = "main-panel-card",
            h4("Cross-validation overview"),
            tags$small("RMSE versus tuning parameter for the selected regression model."),
            withSpinner(plotOutput("cv_plot", height = "360px"))
          )
        ),
        
        tabPanel(
          "📊 Variable importance",
          br(),
          div(
            class = "main-panel-card",
            h4("Top 20 most influential variables"),
            tags$small("Variable importance is based on model coefficients (LASSO) or PLS variable importance measures."),
            withSpinner(plotOutput("vip_plot", height = "400px")),
            br(),
            downloadButton("download_vip_plot", "Download variable importance plot")
          )
        ),
        
        tabPanel(
          "📌 LASSO coefficients",
          br(),
          div(
            class = "main-panel-card",
            h4("Non-zero LASSO coefficients at optimal λ"),
            tags$small("Coefficients are shown for the LASSO model at the cross-validation–selected penalty parameter."),
            br(), br(),
            withSpinner(DTOutput("lasso_coef_table"))
          )
        ),
        
        tabPanel(
          "🔬 Model comparison",
          br(),
          div(
            class = "main-panel-card",
            fluidRow(
              column(
                6,
                h4("LASSO predictions"),
                withSpinner(DTOutput("lasso_pred_table"))
              ),
              column(
                6,
                h4("PLS predictions"),
                withSpinner(DTOutput("pls_pred_table"))
              )
            )
          )
        ),
        
        tabPanel(
          "📖 Interpretation guide",
          br(),
          div(
            class = "main-panel-card",
            h4("How to interpret predictions"),
            p("Values represent predicted percent Robusta adulteration in instant Arabica coffee."),
            tags$ul(
              tags$li("Below 10%: Likely pure or low-level adulteration."),
              tags$li("10–20%: Suspicious or moderate adulteration."),
              tags$li(">20%: High likelihood of significant adulteration.")
            )
          )
        ),
        
        tabPanel(
          "📘 Model info",
          br(),
          div(
            class = "main-panel-card",
            h4("Model training details"),
            p("The regression models were trained using raw near-infrared hyperspectral imaging (HSI) spectra (935–1720 nm) from instant Arabica coffee samples adulterated with Robusta."),
            h5("LASSO regression"),
            tags$ul(
              tags$li("Type: Linear model with L1 regularization."),
              tags$li("Purpose: Feature selection and sparsity."),
              tags$li("Trained on: Raw spectral data."),
              tags$li("Optimized using 10-fold cross-validation.")
            ),
            h5("PLS regression"),
            tags$ul(
              tags$li("Type: Partial least squares regression."),
              tags$li("Purpose: Handle collinearity and reduce dimensionality."),
              tags$li("Trained on: Raw spectral data."),
              tags$li("Optimized for number of latent components via cross-validation.")
            )
          )
        ),
        
        tabPanel(
          "🎨 Theme",
          br(),
          div(
            class = "main-panel-card",
            h4("Customize interface"),
            themeSelector()
          )
        ),
        
        tabPanel(
          "🧪 Example file preview",
          br(),
          div(
            class = "main-panel-card",
            h4("Example dataset preview (from test_samples_instant_coffee.xlsx)"),
            withSpinner(DTOutput("sample_preview"))
          )
        ),
        
        tabPanel(
          "ℹ️ About",
          br(),
          div(
            class = "main-panel-card",
            h4("About SpectraQuant"),
            p("SpectraQuant is an interactive prototype that uses NIR-HS and machine-learning regression models (LASSO, PLS) to estimate Robusta adulteration in instant Arabica coffee."),
            p("Upload your spectral file and select a model to obtain predictions. Use the visualisations and importance plots for interpretation and reporting."),
            p("The app is intended for demonstration, research, and teaching purposes. It is not yet validated for routine quality control."),
            hr(),
            h5("Contact"),
            p("Email: malaviderick@gmail.com"),
            p("GitHub: ",
              a(href = "https://github.com/DNMalavi", "github.com/DNMalavi")),
            p("LinkedIn: ",
              a(href = "https://www.linkedin.com/in/derick-malavi-64742643/",
                "linkedin.com/in/derick-malavi"))
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Raw preview of uploaded file
  input_preview <- reactiveVal(NULL)
  
  # Raw spectra (numeric matrix aligned to X_column_names)
  spectra_raw <- reactiveVal(NULL)
  
  # ===== Helper: LASSO coefficient data (non-zero, with wavelength from X935) =====
  lasso_coef_data <- reactive({
    glmnet_fit  <- lasso_model$finalModel
    lambda_best <- lasso_model$bestTune$lambda
    
    coefs <- as.matrix(coef(glmnet_fit, s = lambda_best))
    df <- data.frame(
      Feature     = rownames(coefs),
      Coefficient = as.numeric(coefs),
      stringsAsFactors = FALSE
    )
    
    # Remove intercept and zero coefficients
    df <- df[df$Feature != "(Intercept)" & df$Coefficient != 0, , drop = FALSE]
    
    # Features are like "X935" -> parse 935 as nm
    wl_candidate <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", df$Feature)))
    
    if (all(!is.na(wl_candidate)) &&
        min(wl_candidate) >= 800 &&
        max(wl_candidate) <= 2500) {
      df$Wavelength_nm <- wl_candidate
    } else {
      df$Wavelength_nm <- NA_real_
    }
    
    df$AbsCoeff <- abs(df$Coefficient)
    df[order(-df$AbsCoeff), , drop = FALSE]
  })
  
  # ===== Data input (align to X_column_names, keep Sample_ID) =====
  data_input <- reactive({
    req(input$file)
    
    ext <- tolower(file_ext(input$file$name))
    
    df_raw <- tryCatch(
      {
        switch(
          ext,
          "csv"  = read_csv(input$file$datapath, show_col_types = FALSE),
          "txt"  = read_delim(input$file$datapath, delim = "\t", show_col_types = FALSE),
          "tsv"  = read_tsv(input$file$datapath, show_col_types = FALSE),
          "xls"  = read_excel(input$file$datapath),
          "xlsx" = read_excel(input$file$datapath),
          stop("Unsupported file type. Please upload CSV, TXT, TSV, XLS, or XLSX.")
        )
      },
      error = function(e) {
        validate(need(FALSE, paste("Failed to read file:", e$message)))
      }
    )
    
    # Store head of raw data for preview
    input_preview(head(df_raw, 5))
    
    # Assume first column is Sample_ID
    sample_ids <- df_raw[[1]]
    spectra_df <- df_raw[, -1, drop = FALSE]
    
    # Keep only numeric columns for spectra
    numeric_cols <- sapply(spectra_df, is.numeric)
    spectra_df   <- spectra_df[, numeric_cols, drop = FALSE]
    
    # If column names missing/blank, try to assign X_column_names
    if (is.null(colnames(spectra_df)) || all(colnames(spectra_df) == "")) {
      if (ncol(spectra_df) == length(X_column_names)) {
        colnames(spectra_df) <- X_column_names
      } else {
        colnames(spectra_df) <- paste0("X", seq_len(ncol(spectra_df)))
      }
    }
    
    # Align spectral columns with X_column_names used in training
    if (ncol(spectra_df) == length(X_column_names)) {
      spectra_df <- spectra_df[, seq_len(length(X_column_names)), drop = FALSE]
      colnames(spectra_df) <- X_column_names
    } else {
      missing_cols <- setdiff(X_column_names, colnames(spectra_df))
      validate(
        need(
          length(missing_cols) == 0,
          paste(
            "Uploaded data are missing",
            length(missing_cols),
            "required spectral variables. Example:",
            paste(head(missing_cols, 5), collapse = ", "), "..."
          )
        )
      )
      spectra_df <- spectra_df[, X_column_names, drop = FALSE]
    }
    
    # Store raw spectra for plotting
    spectra_raw(spectra_df)
    
    # Attach Sample_ID as attribute
    attr(spectra_df, "Sample_ID") <- sample_ids
    spectra_df
  })
  
  # ===== Variable importance data (shared by plot and download) =====
  varimp_df <- reactive({
    model <- if (input$model_type == "LASSO") lasso_model else pls_model
    
    if (input$model_type == "LASSO") {
      glmnet_fit  <- model$finalModel
      lambda_best <- model$bestTune$lambda
      
      coefs <- as.matrix(coef(glmnet_fit, s = lambda_best))
      imp   <- data.frame(
        Feature    = rownames(coefs),
        Importance = as.numeric(coefs),
        stringsAsFactors = FALSE
      )
      # Remove intercept
      imp <- imp[imp$Feature != "(Intercept)", , drop = FALSE]
      
    } else {
      imp <- tryCatch(
        {
          vi <- varImp(model)
          df <- as.data.frame(vi$importance)
          df$Feature <- rownames(df)
          colnames(df)[1] <- "Importance"
          df
        },
        error = function(e) NULL
      )
      if (is.null(imp)) return(NULL)
    }
    
    # Order by absolute importance and keep top 20
    imp$AbsImportance <- abs(imp$Importance)
    imp <- imp[order(-imp$AbsImportance), , drop = FALSE]
    imp[1:min(20, nrow(imp)), , drop = FALSE]
  })
  
  # ===== Predictions based on selected model =====
  predictions <- eventReactive(input$predict, {
    df <- data_input()
    sample_ids <- attr(df, "Sample_ID")
    
    pred <- if (input$model_type == "LASSO") {
      predict(lasso_model, newdata = df)
    } else {
      predict(pls_model, newdata = df)
    }
    
    data.frame(
      Sample_ID              = sample_ids,
      Predicted_Adulteration = round(as.numeric(pred), 2),
      stringsAsFactors = FALSE
    )
  })
  
  # ===== Outputs =====
  
  # Uploaded file metadata
  output$file_info <- renderPrint({
    req(input$file)
    list(
      Name        = input$file$name,
      Type        = input$file$type,
      Size_kB     = round(input$file$size / 1024, 2),
      Uploaded_at = Sys.time()
    )
  })
  
  # Data preview (raw head)
  output$data_preview <- renderDT({
    req(input_preview())
    datatable(input_preview(), options = list(scrollX = TRUE))
  })
  
  # Prediction table
  output$predictions <- renderDT({
    req(predictions())
    datatable(predictions(), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Download predictions
  output$downloadData <- downloadHandler(
    filename = function() {
      paste0(input$filename, "_", tolower(input$model_type), "_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(predictions(), file, row.names = FALSE)
    }
  )
  
  # Download example template (using real test_samples_instant_coffee.xlsx)
  output$downloadTemplate <- downloadHandler(
    filename = function() { "Instant_Coffee_Test_Samples.csv" },
    content = function(file) {
      sample_raw <- read_excel("test_samples_instant_coffee.xlsx")
      # assume first column is Sample_ID; drop any non-numeric spectral columns
      sample_ids  <- sample_raw[[1]]
      spectra_df  <- sample_raw[, -1, drop = FALSE]
      numeric_cols <- sapply(spectra_df, is.numeric)
      spectra_df   <- spectra_df[, numeric_cols, drop = FALSE]
      
      # Align to X_column_names
      missing_cols <- setdiff(X_column_names, colnames(spectra_df))
      if (length(missing_cols) == 0) {
        spectra_df <- spectra_df[, X_column_names, drop = FALSE]
      } else {
        # if names don't match, force names but keep structure (assumes correct order)
        if (ncol(spectra_df) == length(X_column_names)) {
          colnames(spectra_df) <- X_column_names
        }
      }
      
      sample_data <- cbind(Sample_ID = sample_ids, spectra_df)
      write.csv(sample_data, file, row.names = FALSE)
    }
  )
  
  # Example file preview (using the same real test file)
  output$sample_preview <- renderDT({
    sample_raw <- read_excel("test_samples_instant_coffee.xlsx")
    sample_ids  <- sample_raw[[1]]
    spectra_df  <- sample_raw[, -1, drop = FALSE]
    numeric_cols <- sapply(spectra_df, is.numeric)
    spectra_df   <- spectra_df[, numeric_cols, drop = FALSE]
    
    # Align to X_column_names if possible
    missing_cols <- setdiff(X_column_names, colnames(spectra_df))
    if (length(missing_cols) == 0) {
      spectra_df <- spectra_df[, X_column_names, drop = FALSE]
    } else if (ncol(spectra_df) == length(X_column_names)) {
      colnames(spectra_df) <- X_column_names
    }
    
    sample_data <- cbind(Sample_ID = sample_ids, spectra_df)
    datatable(sample_data, options = list(scrollX = TRUE, pageLength = 5))
  })
  
  # Distribution of predicted adulteration (ggplot, grid-free, black text)
  output$prediction_plot <- renderPlot({
    req(predictions())
    df <- predictions()
    
    ggplot(df, aes(x = Predicted_Adulteration)) +
      geom_histogram(bins = 10, fill = "steelblue", color = "black") +
      theme_bw(base_size = 16) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title       = element_text(color = "black"),
        axis.text        = element_text(color = "black")
      ) +
      labs(
        title = "Adulteration level distribution",
        x     = "% Robusta adulteration",
        y     = "Number of samples"
      )
  })
  
  # Spectra viewer (raw only)
  output$spectra_plot <- renderPlot({
    req(spectra_raw())
    mat <- as.matrix(spectra_raw())
    
    # Limit number of spectra to avoid clutter
    n_samp <- min(30, nrow(mat))
    mat <- mat[seq_len(n_samp), , drop = FALSE]
    
    df_long <- as.data.frame(mat)
    df_long$Sample <- factor(paste0("S", seq_len(nrow(df_long))))
    
    df_long <- df_long %>%
      pivot_longer(
        cols = -Sample,
        names_to = "Var",
        values_to = "Intensity"
      )
    
    # Try to map variable names to numeric wavelengths
    wl_num <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", colnames(mat))))
    if (length(wl_num) == ncol(mat) && !any(is.na(wl_num))) {
      wl_map <- data.frame(Var = colnames(mat), Wavelength = wl_num)
      df_long <- df_long %>%
        left_join(wl_map, by = "Var")
      x_var <- "Wavelength"
      x_lab <- "Wavelength (nm)"
    } else {
      df_long$WavelengthIndex <- as.numeric(factor(df_long$Var))
      x_var <- "WavelengthIndex"
      x_lab <- "Wavelength index"
    }
    
    ggplot(df_long, aes_string(x = x_var, y = "Intensity", group = "Sample", color = "Sample")) +
      geom_line(alpha = 0.7) +
      theme_bw(base_size = 13) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title       = element_text(color = "black"),
        axis.text        = element_text(color = "black"),
        legend.position  = "none"
      ) +
      labs(
        title = "Sample spectra (raw NIR-HSI)",
        x     = x_lab,
        y     = "Intensity"
      )
  })
  
  # Cross-validation plots for LASSO and PLS (grid-free, centered, black text)
  output$cv_plot <- renderPlot({
    if (input$model_type == "LASSO") {
      df   <- lasso_model$results
      best <- df[which.min(df$RMSE), ]
      
      ggplot(df, aes(x = lambda, y = RMSE)) +
        geom_line(color = "steelblue", linewidth = 1) +
        geom_point(color = "darkred", size = 2) +
        geom_vline(xintercept = best$lambda, linetype = "dashed", color = "red") +
        annotate(
          "text",
          x     = best$lambda,
          y     = best$RMSE,
          label = paste("Best RMSE:", round(best$RMSE, 2)),
          vjust = -1, hjust = 0, size = 4.5, color = "red"
        ) +
        scale_x_log10() +
        theme_bw(base_size = 16) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = "LASSO cross-validation curve",
          x     = expression(lambda),
          y     = "RMSE"
        )
    } else {
      df   <- pls_model$results
      best <- df[which.min(df$RMSE), ]
      
      ggplot(df, aes(x = ncomp, y = RMSE)) +
        geom_line(color = "darkgreen", linewidth = 1) +
        geom_point(color = "black", size = 2) +
        geom_vline(xintercept = best$ncomp, linetype = "dashed", color = "red") +
        annotate(
          "text",
          x     = best$ncomp,
          y     = best$RMSE,
          label = paste("Best RMSE:", round(best$RMSE, 2)),
          vjust = -1, hjust = 0, size = 4.5, color = "red"
        ) +
        theme_bw(base_size = 16) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = "PLS cross-validation curve",
          x     = "Number of components",
          y     = "RMSE"
        )
    }
  })
  
  # Variable importance plot (viridis colours, grid-free, centered, black text)
  output$vip_plot <- renderPlot({
    df <- varimp_df()
    req(df)
    
    ggplot(df, aes(x = reorder(Feature, AbsImportance), y = Importance, fill = AbsImportance)) +
      geom_col(color = "black") +
      coord_flip() +
      scale_fill_viridis(option = "C", direction = -1) +
      theme_bw(base_size = 16) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title       = element_text(color = "black"),
        axis.text        = element_text(color = "black")
      ) +
      labs(
        title = "Top variable importances",
        x     = "Feature",
        y     = "Importance",
        fill  = "|Importance|"
      )
  })
  
  # Download variable importance plot
  output$download_vip_plot <- downloadHandler(
    filename = function() {
      paste0("spectraquant_varimp_", tolower(input$model_type), "_", Sys.Date(), ".png")
    },
    content = function(file) {
      df <- varimp_df()
      req(df)
      
      p <- ggplot(df, aes(x = reorder(Feature, AbsImportance), y = Importance, fill = AbsImportance)) +
        geom_col(color = "black") +
        coord_flip() +
        scale_fill_viridis(option = "C", direction = -1) +
        theme_bw(base_size = 16) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = "Top variable importances",
          x     = "Feature",
          y     = "Importance",
          fill  = "|Importance|"
        )
      
      ggsave(file, p, width = 7, height = 5, dpi = 300)
    }
  )
  
  # LASSO coefficients table (non-zero)
  output$lasso_coef_table <- renderDT({
    df <- lasso_coef_data()
    if (nrow(df) == 0) return(NULL)
    out <- df[, c("Feature", "Wavelength_nm", "Coefficient", "AbsCoeff"), drop = FALSE]
    colnames(out) <- c("Feature", "Wavelength (nm)", "Coefficient", "|Coefficient|")
    datatable(
      out,
      options = list(pageLength = 20, scrollX = TRUE),
      rownames = FALSE
    )
  })
  
  # Model comparison tables
  output$lasso_pred_table <- renderDT({
    req(data_input())
    df   <- data_input()
    ids  <- attr(df, "Sample_ID")
    preds <- predict(lasso_model, newdata = df)
    out  <- data.frame(
      Sample_ID       = ids,
      Predicted_LASSO = round(as.numeric(preds), 2),
      stringsAsFactors = FALSE
    )
    datatable(out, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$pls_pred_table <- renderDT({
    req(data_input())
    df   <- data_input()
    ids  <- attr(df, "Sample_ID")
    preds <- predict(pls_model, newdata = df)
    out  <- data.frame(
      Sample_ID     = ids,
      Predicted_PLS = round(as.numeric(preds), 2),
      stringsAsFactors = FALSE
    )
    datatable(out, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Reset app
  observeEvent(input$reset, {
    input_preview(NULL)
    spectra_raw(NULL)
    output$data_preview       <- renderDT(NULL)
    output$predictions        <- renderDT(NULL)
    output$prediction_plot    <- renderPlot(NULL)
    output$vip_plot           <- renderPlot(NULL)
    output$lasso_pred_table   <- renderDT(NULL)
    output$pls_pred_table     <- renderDT(NULL)
    output$file_info          <- renderPrint(NULL)
    output$cv_plot            <- renderPlot(NULL)
    output$lasso_coef_table   <- renderDT(NULL)
    output$sample_preview     <- renderDT(NULL)
    output$spectra_plot       <- renderPlot(NULL)
    updateFileInput(session, "file", NULL)
  })
}

# Launch App
shinyApp(ui = ui, server = server)

####______________________________END______________________________________#####
