# ============================================================
# SpectraVision: HSI Instant Coffee Classifier (Cost-Sensitive)
#
# Description:
# This Shiny app classifies instant coffee samples using
# Near-Infrared Hyperspectral Imaging (HSI, ~900–1700 nm)
# and cost-sensitive machine learning models.
#
# Features:
# - MSC + Savitzky–Golay 2nd derivative preprocessing
# - PCA-based input space for classification
# - Cost-sensitive logistic regression and ranger models
# - Spectra visualisation (raw vs preprocessed)
# - PCA score plots with selectable components
# - PCA loading plots and top-loading wavelengths
# - Variable importance (logistic regression only)
# - Prediction tables, class distributions, and sample data
#
# Note:
# This application is a research-stage prototype intended for
# demonstration and exploratory analysis only. It is not
# validated for routine quality control.
#
# Author: Derick Malavi
# Date: June 2025
# ============================================================

library(shiny)
library(ggplot2)
library(ranger)
library(dplyr)
library(readr)
library(mdatools)
library(shinythemes)
library(bslib)
library(DT)
library(shinycssloaders)
library(shinyjs)
library(readxl)
library(tools)
library(caret)
library(tidyr)

# ===========================
# Load models and PCA object
# ===========================

PCA_object     <- readRDS("pca_Preprocess.rds")
ranger_model   <- readRDS("fit_ranger_cost_sensitive.rds")
logreg_model   <- readRDS("fit_logistic_reg_cost_sensitive.rds")
X_column_names <- readRDS("X_column_names.rds")  # e.g. "X935", "X940", ...

# ===========================
# Preprocessing helper
# ===========================

apply_msc_sg2 <- function(df) {
  m  <- as.matrix(df)
  m1 <- prep.msc(m)
  m2 <- prep.savgol(m1, width = 13, porder = 2, dorder = 2)
  as.data.frame(m2)
}

# ===========================
# UI
# ===========================

ui <- fluidPage(
  useShinyjs(),
  theme = bs_theme(version = 5),
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f5f7f8;
      }
      .app-header {
        background: linear-gradient(to right, #111827, #1f2937, #4f46e5);
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
      .sidebarPanel {
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,0.08);
      }
      .main-panel-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 12px 16px;
        box-shadow: 0 4px 16px rgba(15,23,42,0.08);
        margin-bottom: 15px;
      }
      .nav-tabs > li > a {
        font-weight: 500;
      }
    "))
  ),
  
  # ------- App header --------
  div(
    class = "app-header",
    div(
      class = "app-header-inner",
      div(
        class = "app-title-main",
        div(
          class = "app-title-row",
          span("🌈 SpectraVision: HSI Instant Coffee Classifier"),
          span(style = "font-size: 13px; opacity: 0.85;", "(Cost-sensitive models)")
        ),
        span(
          class = "app-subtitle",
          HTML("Near-Infrared Hyperspectral Imaging · Chemometrics · Cost-sensitive machine learning")
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
      h4("1. Upload data"),
      tags$small("Upload a spectral dataset with one row per sample and wavelength variables in columns."),
      fileInput(
        "test_file",
        "Upload spectral data file",
        accept = c(".csv", ".txt", ".tsv", ".xls", ".xlsx")
      ),
      hr(),
      h4("2. Model configuration"),
      selectInput(
        "model_choice",
        "Choose classifier:",
        choices = c(
          "Cost-sensitive logistic regression (PCA)" = "logreg",
          "Cost-sensitive ranger (PCA)"             = "ranger"
        )
      ),
      selectInput(
        "pca_x",
        "PCA X-axis:",
        choices = c("PC1" = 1, "PC2" = 2, "PC3" = 3),
        selected = 1
      ),
      selectInput(
        "pca_y",
        "PCA Y-axis:",
        choices = c("PC1" = 1, "PC2" = 2, "PC3" = 3),
        selected = 2
      ),
      radioButtons(
        "bar_type",
        "Prediction bar scale:",
        choices = c("Counts" = "count", "Percentages" = "percent"),
        selected = "count",
        inline = TRUE
      ),
      hr(),
      h4("3. Appearance and export"),
      selectInput(
        "theme_choice",
        "Theme:",
        choices = c(
          "Default"   = "default",
          "Cerulean"  = "cerulean",
          "Cosmo"     = "cosmo",
          "Darkly"    = "darkly",
          "Flatly"    = "flatly",
          "Journal"   = "journal",
          "Litera"    = "litera",
          "Lumen"     = "lumen",
          "Minty"     = "minty",
          "Pulse"     = "pulse",
          "Sandstone" = "sandstone",
          "Spacelab"  = "spacelab",
          "United"    = "united",
          "Yeti"      = "yeti"
        ),
        selected = "cosmo"
      ),
      textInput("filename", "Custom filename", "SpectraVision_predictions"),
      br(),
      actionButton("predict_btn", "🚀 Run prediction", class = "btn btn-success btn-block"),
      actionButton("reset_btn",   "🔄 Reset app",      class = "btn btn-warning btn-block"),
      hr(),
      downloadButton("download_preds",  "Download predictions", class = "btn btn-primary btn-block")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel(
          "📘 User guide",
          br(),
          div(
            class = "main-panel-card",
            h3("How to use SpectraVision (cost-sensitive classifier)"),
            tags$ol(
              tags$li("Prepare your spectral file so that the first column contains sample IDs and the remaining columns contain NIR-HSI wavelengths matching the training structure."),
              tags$li("Upload the file using the \"Upload spectral data file\" control in the sidebar."),
              tags$li("Select a classifier (logistic regression or cost-sensitive ranger) and choose the PCA axes for visualisation."),
              tags$li("Click \"Run prediction\" to generate class predictions."),
              tags$li("Explore the tabs for data overview, spectra, PCA scores, PCA loadings, logistic regression variable importance, and detailed prediction tables."),
              tags$li("Download the prediction table for reporting or further analysis.")
            ),
            p("The models distinguish authentic instant Arabica coffee from adulterated samples based on NIR-HSI spectra and cost-sensitive learning.")
          )
        ),
        
        tabPanel(
          "📊 Overview",
          br(),
          div(class = "main-panel-card",
              h4("Uploaded data preview"),
              tags$small("First five rows of the uploaded file are shown (raw input)."),
              withSpinner(DTOutput("preview_data"))
          ),
          div(class = "main-panel-card",
              h4("Prediction summary"),
              tags$small("Summary of predicted classes across all uploaded samples."),
              withSpinner(DTOutput("results_summary")),
              br(),
              withSpinner(plotOutput("bar_distribution", height = "360px"))
          )
        ),
        
        tabPanel(
          "📡 Spectra",
          br(),
          div(
            class = "main-panel-card",
            h4("Spectral profiles"),
            tags$small("Visualise raw spectra or MSC + Savitzky–Golay 2nd derivative."),
            radioButtons(
              "spectra_view",
              "Spectra shown as:",
              choices = c(
                "Raw (unpreprocessed)"               = "raw",
                "MSC + SG 2nd derivative (width 13)" = "MSC_SG2"
              ),
              selected = "raw",
              inline = TRUE
            ),
            withSpinner(plotOutput("spectra_plot", height = "360px"))
          )
        ),
        
        tabPanel(
          "📈 PCA scores",
          br(),
          div(
            class = "main-panel-card",
            h4("PCA score plot"),
            tags$small("PCA is applied to MSC + SG preprocessed spectra and used as input to the classifiers."),
            withSpinner(plotOutput("pca_dynamic_plot", height = "360px")),
            br(),
            downloadButton("download_pca_scores", "Download PCA score plot")
          )
        ),
        
        tabPanel(
          "📉 PCA loadings",
          br(),
          div(
            class = "main-panel-card",
            h4("PCA loading plot"),
            tags$small("Loadings show how strongly each wavelength contributes to the selected principal component(s)."),
            selectInput(
              "loading_pcs",
              "Select PCs to display:",
              choices = NULL,
              multiple = TRUE
            ),
            withSpinner(plotOutput("pca_loadings_plot", height = "360px")),
            br(),
            downloadButton("download_pca_loadings_plot", "Download loadings plot"),
            br(), br(),
            h4("Top loading wavelengths"),
            tags$small("Top wavelengths (by absolute loading) for each selected principal component."),
            withSpinner(DTOutput("pca_loadings_table"))
          )
        ),
        
        tabPanel(
          "⭐ Variable importance",
          br(),
          div(
            class = "main-panel-card",
            h4("Top 20 most important variables (logistic regression)"),
            tags$small("Variable importance is computed only for the cost-sensitive logistic regression classifier."),
            withSpinner(plotOutput("varimp_plot", height = "400px")),
            br(),
            downloadButton("download_varimp", "Download variable importance plot"),
            br(), br(),
            withSpinner(DTOutput("varimp_table"))
          )
        ),
        
        tabPanel(
          "📋 Predictions",
          br(),
          div(
            class = "main-panel-card",
            h4("Prediction details"),
            tags$small("Predicted class for each sample."),
            withSpinner(DTOutput("prediction_table")),
            br(),
            verbatimTextOutput("class_summary")
          )
        ),
        
        tabPanel(
          "🧪 Sample data",
          br(),
          div(
            class = "main-panel-card",
            h4("Example test set"),
            p("Download an example instant coffee test set derived from real NIR-HSI measurements. True labels have been removed so that the file can be used as a blind test."),
            downloadButton("download_sample", "Download sample CSV")
          )
        ),
        
        tabPanel(
          "⚙️ Preprocessing info",
          br(),
          div(
            class = "main-panel-card",
            h3("Preprocessing pipeline"),
            p("Spectra are preprocessed using the same pipeline that was used for training the cost-sensitive models:"),
            tags$ul(
              tags$li("Multiplicative scatter correction (MSC): Corrects multiplicative scatter and path-length effects."),
              tags$li("Savitzky–Golay derivative: Smoothing and second-order derivative (window width 13, polynomial order 2, derivative order 2) to enhance subtle spectral features."),
              tags$li("Principal component analysis (PCA): Used to construct a low-dimensional input space for the classifiers.")
            )
          )
        ),
        
        tabPanel(
          "📝 Feedback",
          br(),
          div(
            class = "main-panel-card",
            h3("Feedback"),
            p("Your feedback helps refine this research prototype."),
            textAreaInput(
              "user_feedback",
              "Comments, suggestions, or issues:",
              "",
              width = "100%",
              height = "120px"
            ),
            actionButton("send_feedback", "📨 Submit feedback", class = "btn btn-primary")
          )
        ),
        
        tabPanel(
          "🔍 Diagnostics",
          br(),
          div(
            class = "main-panel-card",
            h3("Diagnostics"),
            p("Basic checks on the uploaded file and model alignment."),
            verbatimTextOutput("diagnostics")
          )
        ),
        
        tabPanel(
          "ℹ️ About",
          br(),
          div(
            class = "main-panel-card",
            h3("About SpectraVision (cost-sensitive prototype)"),
            p("SpectraVision demonstrates how near-infrared hyperspectral imaging and cost-sensitive machine learning can be used to classify instant coffee samples."),
            tags$ul(
              tags$li("Models: cost-sensitive logistic regression and ranger, trained on PCA-transformed NIR-HSI spectra."),
              tags$li("Spectral range: approximately 900–1700 nm from instant Arabica coffee adulterated with Robusta."),
              tags$li("Preprocessing: MSC followed by Savitzky–Golay second-order derivatives, with PCA used as the model input space."),
              tags$li("Outputs: class predictions, prediction summaries, PCA score plots, PCA loading plots, spectra visualisation, and logistic regression variable importance.")
            ),
            p("This application is a research-stage prototype intended for demonstration and exploratory analysis only and is not validated for routine quality control use.")
          )
        ),
        
        tabPanel(
          "📩 Contact",
          br(),
          div(
            class = "main-panel-card",
            h3("Developer"),
            p("Name: Derick Malavi"),
            p("Email: ", a(href = "mailto:malaviderick@gmail.com", "malaviderick@gmail.com")),
            p("GitHub: ", a(href = "https://github.com/DNMalavi", "github.com/DNMalavi")),
            p("LinkedIn: ", a(href = "https://www.linkedin.com/in/derick-malavi-64742643/", "linkedin.com/in/derick-malavi"))
          )
        )
      )
    )
  )
)

# ===========================
# Server
# ===========================

server <- function(input, output, session) {
  
  pred_results    <- reactiveVal()
  input_preview   <- reactiveVal()
  pca_scores      <- reactiveVal()
  pca_predictions <- reactiveVal()
  spectra_raw     <- reactiveVal()
  
  # ---- dynamic theme ----
  observe({
    req(input$theme_choice)
    if (input$theme_choice == "default") {
      session$setCurrentTheme(bs_theme(version = 5))
    } else {
      session$setCurrentTheme(bs_theme(bootswatch = input$theme_choice))
    }
  })
  
  # ---- dynamic PCA PC choices based on PCA_object ----
  observe({
    if (!is.null(PCA_object) && !is.null(PCA_object$rotation)) {
      ncomp <- ncol(PCA_object$rotation)
      if (ncomp >= 1) {
        pc_choices <- setNames(seq_len(ncomp), paste0("PC", seq_len(ncomp)))
        updateSelectInput(session, "pca_x",       choices = pc_choices, selected = min(1, ncomp))
        updateSelectInput(session, "pca_y",       choices = pc_choices, selected = min(2, ncomp))
        updateSelectInput(session, "loading_pcs", choices = pc_choices, selected = 1)
      }
    }
  })
  
  # ---- run prediction ----
  observeEvent(input$predict_btn, {
    req(input$test_file)
    
    ext <- tolower(file_ext(input$test_file$name))
    
    new_data <- tryCatch(
      {
        switch(
          ext,
          "csv"  = read_csv(input$test_file$datapath, show_col_types = FALSE),
          "txt"  = read_delim(input$test_file$datapath, delim = "\t", show_col_types = FALSE),
          "tsv"  = read_tsv(input$test_file$datapath, show_col_types = FALSE),
          "xls"  = read_excel(input$test_file$datapath),
          "xlsx" = read_excel(input$test_file$datapath),
          stop("Unsupported file type. Please upload CSV, TXT, TSV, XLS, or XLSX.")
        )
      },
      error = function(e) {
        showNotification(paste("Failed to read file:", e$message), type = "error")
        return(NULL)
      }
    )
    req(new_data)
    
    # Store raw preview
    input_preview(head(new_data, 5))
    output$preview_data <- renderDT({
      datatable(input_preview(), options = list(scrollX = TRUE))
    })
    
    # Assume first column is Sample_ID, rest are spectra
    sample_ids <- new_data[[1]]
    spectra_df <- new_data[, -1, drop = FALSE]
    
    # Use only numeric columns as spectra
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
    
    # Align with training X_column_names
    if (ncol(spectra_df) == length(X_column_names)) {
      spectra_df <- spectra_df[, seq_len(length(X_column_names)), drop = FALSE]
      colnames(spectra_df) <- X_column_names
    } else {
      missing_cols <- setdiff(X_column_names, colnames(spectra_df))
      if (length(missing_cols) > 0) {
        showNotification(
          paste(
            "Uploaded data are missing",
            length(missing_cols),
            "required spectral variables. Example:",
            paste(head(missing_cols, 5), collapse = ", "), "..."
          ),
          type = "error",
          duration = 8
        )
        return(NULL)
      }
      spectra_df <- spectra_df[, X_column_names, drop = FALSE]
    }
    
    # Store raw spectra
    spectra_raw(spectra_df)
    
    # Preprocess (MSC + SG 2nd derivative)
    processed_df <- apply_msc_sg2(spectra_df)
    
    # PCA projection using stored PCA_object
    if (!is.null(PCA_object)) {
      X_pca <- predict(PCA_object, processed_df)
      X_pca <- as.data.frame(X_pca)
      colnames(X_pca) <- paste0("PC", seq_len(ncol(X_pca)))
      pca_scores(X_pca)
    } else {
      X_pca <- processed_df
      pca_scores(NULL)
    }
    
    # Select model and predict
    predictions <- switch(
      input$model_choice,
      "ranger" = predict(ranger_model, newdata = X_pca),
      "logreg" = predict(logreg_model, newdata = X_pca),
      stop("Unknown model choice.")
    )
    
    result_df <- data.frame(
      Sample_ID  = sample_ids,
      Prediction = predictions,
      stringsAsFactors = FALSE
    )
    
    pred_results(result_df)
    pca_predictions(predictions)
    
    # ---- outputs ----
    output$prediction_table <- renderDT({
      datatable(result_df, options = list(scrollX = TRUE))
    })
    
    output$class_summary <- renderPrint({
      table(result_df$Prediction)
    })
    
    # Summary table
    output$results_summary <- renderDT({
      as.data.frame(table(result_df$Prediction)) |>
        dplyr::rename(Class = Var1, Count = Freq)
    })
    
    # Bar distribution (counts or %)
    output$bar_distribution <- renderPlot({
      df <- result_df
      
      if (input$bar_type == "percent") {
        df_plot <- df %>%
          count(Prediction) %>%
          mutate(Percent = n / sum(n) * 100)
        
        ggplot(df_plot, aes(x = Prediction, y = Percent, fill = Prediction)) +
          geom_col(width = 0.7, color = "black") +
          scale_fill_viridis_d(option = "turbo", direction = 1) +
          theme_bw(base_size = 14) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
            axis.title       = element_text(color = "black"),
            axis.text        = element_text(color = "black")
          ) +
          labs(
            title = "Distribution of predicted classes (percent)",
            x     = "Predicted class",
            y     = "Percentage of samples"
          )
      } else {
        ggplot(df, aes(x = Prediction, fill = Prediction)) +
          geom_bar(width = 0.7, color = "black") +
          scale_fill_viridis_d(option = "turbo", direction = 1) +
          theme_bw(base_size = 14) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
            axis.title       = element_text(color = "black"),
            axis.text        = element_text(color = "black")
          ) +
          labs(
            title = "Distribution of predicted classes (counts)",
            x     = "Predicted class",
            y     = "Number of samples"
          )
      }
    })
    
    # PCA scores plot
    output$pca_dynamic_plot <- renderPlot({
      req(pca_scores())
      df <- as.data.frame(pca_scores())
      if (ncol(df) < 2) return(NULL)
      
      df$Prediction <- pca_predictions()
      
      x_pc <- paste0("PC", input$pca_x)
      y_pc <- paste0("PC", input$pca_y)
      
      df_plot <- df
      df_plot$x_val <- df_plot[[x_pc]]
      df_plot$y_val <- df_plot[[y_pc]]
      
      ggplot(df_plot, aes(x = x_val, y = y_val, color = Prediction)) +
        geom_point(size = 3, alpha = 0.9) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA score plot:", x_pc, "vs", y_pc),
          x     = x_pc,
          y     = y_pc,
          color = "Prediction"
        )
    })
    
    showNotification("Prediction complete.", type = "message")
  })
  
  # ========= Spectra plot (raw vs MSC+SG2) =========
  output$spectra_plot <- renderPlot({
    req(spectra_raw())
    mat <- spectra_raw()
    
    if (input$spectra_view == "MSC_SG2") {
      mat <- as.matrix(apply_msc_sg2(mat))
    } else {
      mat <- as.matrix(mat)  # raw
    }
    
    # Limit number of spectra for visual clarity
    n_samp <- min(30, nrow(mat))
    mat <- mat[seq_len(n_samp), , drop = FALSE]
    
    df_long <- as.data.frame(mat)
    df_long$Sample <- factor(paste0("S", seq_len(nrow(df_long))))
    
    df_long <- df_long %>%
      pivot_longer(
        cols      = -Sample,
        names_to  = "Var",
        values_to = "Intensity"
      )
    
    wl_num <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", colnames(mat))))
    if (length(wl_num) == ncol(mat) && !any(is.na(wl_num))) {
      wl_map <- data.frame(Var = colnames(mat), Wavelength = wl_num)
      df_long <- df_long %>%
        left_join(wl_map, by = "Var")
      x_lab <- "Wavelength (nm)"
      
      ggplot(df_long, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
        geom_line(alpha = 0.6) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
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
          title = "Sample spectra",
          x     = x_lab,
          y     = "Intensity"
        )
    } else {
      df_long$WavelengthIndex <- as.numeric(gsub("X", "", df_long$Var))
      ggplot(df_long, aes(x = WavelengthIndex, y = Intensity, group = Sample, color = Sample)) +
        geom_line(alpha = 0.6) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
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
          title = "Sample spectra",
          x     = "Wavelength index",
          y     = "Intensity"
        )
    }
  })
  
  # ========= PCA loadings (with max points) =========
  pca_loadings_data <- reactive({
    req(PCA_object, PCA_object$rotation)
    pcs <- input$loading_pcs
    req(pcs)
    
    loadings <- PCA_object$rotation
    pcs_idx  <- as.integer(pcs)
    pcs_idx  <- pcs_idx[pcs_idx >= 1 & pcs_idx <= ncol(loadings)]
    req(length(pcs_idx) > 0)
    
    pc_names <- colnames(loadings)[pcs_idx]
    
    df <- as.data.frame(loadings[, pc_names, drop = FALSE])
    df$Variable <- rownames(loadings)
    
    wl <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", df$Variable)))
    if (length(wl) == nrow(df) && !all(is.na(wl))) {
      df$Wavelength_nm <- wl
    } else {
      df$Wavelength_nm <- NA_real_
    }
    
    df_long <- df %>%
      pivot_longer(
        cols = all_of(pc_names),
        names_to = "PC",
        values_to = "Loading"
      )
    
    df_long$AbsLoading <- abs(df_long$Loading)
    df_long
  })
  
  output$pca_loadings_plot <- renderPlot({
    df_long <- pca_loadings_data()
    req(nrow(df_long) > 0)
    
    # Top max |loading| per PC
    top_points <- df_long %>%
      group_by(PC) %>%
      slice_max(order_by = AbsLoading, n = 1, with_ties = FALSE) %>%
      ungroup()
    
    if (!all(is.na(df_long$Wavelength_nm))) {
      ggplot(df_long, aes(x = Wavelength_nm, y = Loading, color = PC)) +
        geom_line(alpha = 0.9) +
        geom_point(
          data = top_points,
          aes(x = Wavelength_nm, y = Loading, color = PC),
          size = 3,
          stroke = 0.8
        ) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = "PCA loadings vs wavelength",
          x     = "Wavelength (nm)",
          y     = "Loading"
        )
    } else {
      df_long$Index <- match(df_long$Variable, unique(df_long$Variable))
      top_points$Index <- match(top_points$Variable, unique(df_long$Variable))
      
      ggplot(df_long, aes(x = Index, y = Loading, color = PC)) +
        geom_line(alpha = 0.9) +
        geom_point(
          data = top_points,
          aes(x = Index, y = Loading, color = PC),
          size = 3,
          stroke = 0.8
        ) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = "PCA loadings (index scale)",
          x     = "Variable index",
          y     = "Loading"
        )
    }
  })
  
  output$pca_loadings_table <- renderDT({
    df_long <- pca_loadings_data()
    if (nrow(df_long) == 0) return(NULL)
    
    df_top <- df_long %>%
      group_by(PC) %>%
      slice_max(order_by = AbsLoading, n = 15, with_ties = FALSE) %>%
      arrange(PC, desc(AbsLoading)) %>%
      ungroup()
    
    datatable(
      df_top[, c("PC", "Variable", "Wavelength_nm", "Loading")],
      options = list(pageLength = 30, scrollX = TRUE),
      rownames = FALSE
    )
  })
  
  # ---- download PCA loadings plot ----
  output$download_pca_loadings_plot <- downloadHandler(
    filename = function() {
      paste0("spectravision_pca_loadings_", Sys.Date(), ".png")
    },
    content = function(file) {
      df_long <- pca_loadings_data()
      if (nrow(df_long) == 0) {
        ggplot() + theme_void()
        ggsave(file, width = 7, height = 5, dpi = 300)
      } else {
        top_points <- df_long %>%
          group_by(PC) %>%
          slice_max(order_by = AbsLoading, n = 1, with_ties = FALSE) %>%
          ungroup()
        
        if (!all(is.na(df_long$Wavelength_nm))) {
          p <- ggplot(df_long, aes(x = Wavelength_nm, y = Loading, color = PC)) +
            geom_line(alpha = 0.9) +
            geom_point(
              data = top_points,
              aes(x = Wavelength_nm, y = Loading, color = PC),
              size = 3,
              stroke = 0.8
            ) +
            scale_color_viridis_d(option = "turbo", direction = 1) +
            theme_bw(base_size = 13) +
            theme(
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
              axis.title       = element_text(color = "black"),
              axis.text        = element_text(color = "black")
            ) +
            labs(
              title = "PCA loadings vs wavelength",
              x     = "Wavelength (nm)",
              y     = "Loading"
            )
        } else {
          df_long$Index <- match(df_long$Variable, unique(df_long$Variable))
          top_points$Index <- match(top_points$Variable, unique(df_long$Variable))
          
          p <- ggplot(df_long, aes(x = Index, y = Loading, color = PC)) +
            geom_line(alpha = 0.9) +
            geom_point(
              data = top_points,
              aes(x = Index, y = Loading, color = PC),
              size = 3,
              stroke = 0.8
            ) +
            scale_color_viridis_d(option = "turbo", direction = 1) +
            theme_bw(base_size = 13) +
            theme(
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
              axis.title       = element_text(color = "black"),
              axis.text        = element_text(color = "black")
            ) +
            labs(
              title = "PCA loadings (index scale)",
              x     = "Variable index",
              y     = "Loading"
            )
        }
        ggsave(file, p, width = 7, height = 5, dpi = 300)
      }
    }
  )
  
  # ========= Variable importance (logistic only) =========
  varimp_data <- reactive({
    req(input$model_choice)
    # Only support varImp for logistic regression
    if (input$model_choice != "logreg") {
      return(data.frame())
    }
    
    model <- logreg_model
    
    imp <- tryCatch(
      {
        vi <- varImp(model, scale = TRUE)
        df <- as.data.frame(vi$importance)
        df$Variable <- rownames(df)
        if ("Overall" %in% names(df)) {
          df$Importance <- df$Overall
        } else {
          score_cols <- setdiff(names(df), "Variable")
          df$Importance <- apply(df[, score_cols, drop = FALSE], 1, mean, na.rm = TRUE)
        }
        df
      },
      error = function(e) NULL
    )
    if (is.null(imp)) return(data.frame())
    
    # Try to parse wavelength
    wl <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", imp$Variable)))
    if (length(wl) == nrow(imp) && !all(is.na(wl))) {
      imp$Wavelength_nm <- wl
    } else {
      imp$Wavelength_nm <- NA_real_
    }
    
    imp$AbsImportance <- abs(imp$Importance)
    imp <- imp[order(-imp$AbsImportance), , drop = FALSE]
    head(imp, 20)
  })
  
  output$varimp_plot <- renderPlot({
    df <- varimp_data()
    if (nrow(df) == 0) {
      # Show a simple message if ranger is selected or no importance
      ggplot() +
        theme_void() +
        geom_text(
          aes(0, 0, label = "Variable importance is available only for the logistic regression model."),
          size = 4
        ) +
        xlim(-1, 1) + ylim(-1, 1)
    } else {
      ggplot(df, aes(x = reorder(Variable, Importance), y = Importance, fill = AbsImportance)) +
        geom_col(width = 0.8, color = "black") +
        coord_flip() +
        scale_fill_viridis_c(option = "turbo", direction = 1) +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          x     = "Variable",
          y     = "Importance",
          title = "Top 20 important variables (logistic regression)"
        )
    }
  })
  
  output$varimp_table <- renderDT({
    df <- varimp_data()
    if (nrow(df) == 0) return(NULL)
    cols <- c("Variable", "Wavelength_nm", "Importance")
    datatable(
      df[, cols],
      options = list(pageLength = 20, scrollX = TRUE),
      rownames = FALSE
    )
  })
  
  # ---- download variable importance plot ----
  output$download_varimp <- downloadHandler(
    filename = function() {
      paste0("spectravision_varimp_logistic_", Sys.Date(), ".png")
    },
    content = function(file) {
      df <- varimp_data()
      if (nrow(df) == 0) {
        ggplot() + theme_void()
        ggsave(file, width = 7, height = 5, dpi = 300)
      } else {
        p <- ggplot(df, aes(x = reorder(Variable, Importance), y = Importance, fill = AbsImportance)) +
          geom_col(width = 0.8, color = "black") +
          coord_flip() +
          scale_fill_viridis_c(option = "turbo", direction = 1) +
          theme_bw(base_size = 13) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
            axis.title       = element_text(color = "black"),
            axis.text        = element_text(color = "black")
          ) +
          labs(
            x     = "Variable",
            y     = "Importance",
            title = "Top 20 important variables (logistic regression)"
          )
        ggsave(file, p, width = 7, height = 5, dpi = 300)
      }
    }
  )
  
  # ---- download PCA scores plot ----
  output$download_pca_scores <- downloadHandler(
    filename = function() {
      paste0("spectravision_pca_scores_", Sys.Date(), ".png")
    },
    content = function(file) {
      req(pca_scores(), pca_predictions())
      df <- as.data.frame(pca_scores())
      df$Prediction <- pca_predictions()
      
      x_pc <- paste0("PC", input$pca_x)
      y_pc <- paste0("PC", input$pca_y)
      
      df_plot <- df
      df_plot$x_val <- df_plot[[x_pc]]
      df_plot$y_val <- df_plot[[y_pc]]
      
      p <- ggplot(df_plot, aes(x = x_val, y = y_val, color = Prediction)) +
        geom_point(size = 3, alpha = 0.9) +
        scale_color_viridis_d(option = "turbo", direction = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title       = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title       = element_text(color = "black"),
          axis.text        = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA score plot:", x_pc, "vs", y_pc),
          x     = x_pc,
          y     = y_pc,
          color = "Prediction"
        )
      ggsave(file, p, width = 7, height = 5, dpi = 300)
    }
  )
  
  # ---- diagnostics ----
  output$diagnostics <- renderPrint({
    tryCatch(
      {
        list(
          "Columns in uploaded file"   = names(input_preview()),
          "First spectral columns"     = head(X_column_names),
          "Number of preview rows"     = if (is.null(input_preview())) 0 else nrow(input_preview()),
          "Spectra loaded (rows)"      = if (is.null(spectra_raw())) 0 else nrow(spectra_raw())
        )
      },
      error = function(e) {
        list("Diagnostics error" = e$message)
      }
    )
  })
  
  # ---- download predictions ----
  output$download_preds <- downloadHandler(
    filename = function() {
      paste0(input$filename, "_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(pred_results(), file, row.names = FALSE)
    }
  )
  
  # ---- sample data from real test file ----
  output$download_sample <- downloadHandler(
    filename = function() { "Instant_Coffee_Test_Samples.csv" },
    content = function(file) {
      # Expect the Excel file to be in the app directory
      sample_file <- "test_samples_instant_coffee.xlsx"
      sample_raw <- read_excel(sample_file)
      
      # Drop true label column if present
      if ("true_class" %in% names(sample_raw)) {
        sample_raw$true_class <- NULL
      }
      
      # Assume first column is Sample_ID, rest spectra
      # Keep only columns that correspond to X_column_names
      spectra_part <- sample_raw[, -1, drop = FALSE]
      numeric_cols <- sapply(spectra_part, is.numeric)
      spectra_part <- spectra_part[, numeric_cols, drop = FALSE]
      
      # Align spectral columns to X_column_names where possible
      common_cols <- intersect(colnames(spectra_part), X_column_names)
      spectra_part <- spectra_part[, common_cols, drop = FALSE]
      
      # If needed, reorder to match X_column_names
      spectra_part <- spectra_part[, intersect(X_column_names, colnames(spectra_part)), drop = FALSE]
      
      # Final sample data for export
      sample_out <- cbind(Sample_ID = sample_raw[[1]], spectra_part)
      write.csv(sample_out, file, row.names = FALSE)
    }
  )
  
  # ---- feedback logging ----
  observeEvent(input$send_feedback, {
    req(input$user_feedback)
    feedback <- data.frame(
      timestamp = Sys.time(),
      feedback  = input$user_feedback,
      stringsAsFactors = FALSE
    )
    feedback_file <- "spectravision_feedback_log.csv"
    if (file.exists(feedback_file)) {
      write.table(
        feedback,
        file      = feedback_file,
        append    = TRUE,
        sep       = ",",
        row.names = FALSE,
        col.names = FALSE
      )
    } else {
      write.table(
        feedback,
        file      = feedback_file,
        append    = FALSE,
        sep       = ",",
        row.names = FALSE,
        col.names = TRUE
      )
    }
    showNotification("Thank you. Your feedback has been saved.", type = "message")
  })
  
  # ---- reset app ----
  observeEvent(input$reset_btn, {
    pred_results(NULL)
    input_preview(NULL)
    pca_scores(NULL)
    pca_predictions(NULL)
    spectra_raw(NULL)
    
    output$prediction_table    <- renderDT(NULL)
    output$class_summary       <- renderPrint(NULL)
    output$results_summary     <- renderDT(NULL)
    output$bar_distribution    <- renderPlot(NULL)
    output$pca_dynamic_plot    <- renderPlot(NULL)
    output$preview_data        <- renderDT(NULL)
    output$varimp_plot         <- renderPlot(NULL)
    output$varimp_table        <- renderDT(NULL)
    output$diagnostics         <- renderPrint(NULL)
    output$pca_loadings_plot   <- renderPlot(NULL)
    output$pca_loadings_table  <- renderDT(NULL)
    
    updateFileInput(session, "test_file", NULL)
    updateTextInput(session, "user_feedback", value = "")
  })
}

shinyApp(ui = ui, server = server)
