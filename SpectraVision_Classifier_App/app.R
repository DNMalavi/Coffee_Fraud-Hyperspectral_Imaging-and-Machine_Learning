# SpectraVision: HSI Instant Coffee Classifier
# -------------------------------------------------------------
# Prototype app for classifying instant Arabica coffee samples using
# near-infrared hyperspectral imaging (HSI) and machine learning.
#
# - Preprocessing: SNV + Savitzky–Golay 2nd derivative (mdatools)
# - Dimensionality reduction: PCA (pre-trained)
# - Classifiers: k-NN and Random Forest (pre-trained caret models)
# - Features:
#     * Upload spectral data (.csv, .txt, .tsv, .xls, .xlsx)
#     * Spectra visualisation: raw vs SNV + SG 2nd derivative
#     * PCA score & loading plots (selectable PCs, not limited to PC1–PC3)
#     * Prediction tables and class distribution plots
#     * Cross-validation plots for RF and k-NN (side by side)
#     * Variable importance (top 20) with viridis gradient
#     * Real example test set from test_samples_instant_coffee.xlsx
#     * Feedback logging and diagnostics
#
# NOTE: Research prototype – not validated for routine QC.

library(shiny)
library(randomForest)
library(class)
library(ggplot2)
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
library(viridis)   # for attractive color scales
library(tidyr)     # for pivot_longer

# ===========================
# Load trained objects
# ===========================
PCA_object       <- readRDS("PCA_Preprocess.rds")
rf_model         <- readRDS("fit_rf.rds")
knn_model        <- readRDS("fit_knn.rds")
X_column_names   <- readRDS("X_column_names.rds")  # expected spectral columns (e.g. X935, X940, ...)

# ===========================
# Helper: preprocessing
# ===========================
apply_snv_sg2 <- function(df) {
  m  <- as.matrix(df)
  m1 <- prep.snv(m)
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
  
  # --- header ---
  div(
    class = "app-header",
    div(
      class = "app-header-inner",
      div(
        class = "app-title-main",
        div(
          class = "app-title-row",
          span("☕ SpectraVision: HSI Instant Coffee Classifier")
        ),
        span(
          class = "app-subtitle",
          HTML("🌈 Near-Infrared Hyperspectral Imaging · Chemometrics · Machine learning")
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
      tags$small("Upload a spectral dataset where each row is a sample and columns are wavelengths (e.g. X935, X940, ...)."),
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
          "k-NN (PCA scores)"          = "knn",
          "Random Forest (PCA scores)" = "rf"
        )
      ),
      radioButtons(
        "preprocess_choice",
        "Spectral preprocessing:",
        choices = c("SNV + SG 2nd derivative" = "snv_sg2"),
        selected = "snv_sg2"
      ),
      helpText("Both models use PCA scores computed from SNV + Savitzky–Golay 2nd derivative spectra."),
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
        selected = "minty"
      ),
      textInput("filename", "Custom filename", "SpectraVision_predictions"),
      br(),
      actionButton("predict_btn", "Run prediction", class = "btn btn-success btn-block"),
      actionButton("reset_btn",   "Reset app",      class = "btn btn-warning btn-block"),
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
            h3("How to use SpectraVision"),
            tags$ol(
              tags$li("Prepare your spectral data so that each row is a sample and each column is a wavelength variable (e.g. X935, X940, ...)."),
              tags$li("Upload the file using the \"Upload spectral data file\" control on the left."),
              tags$li("Choose the classifier (k-NN or Random Forest)."),
              tags$li("Click \"Run prediction\" to generate class predictions."),
              tags$li("Explore the tabs for spectra, PCA scores & loadings, cross-validation, variable importance, and predictions."),
              tags$li("Download prediction tables and the example test set for demonstration.")
            ),
            p("The models classify instant Arabica coffee samples based on NIR-HSI spectra and were trained on medium-roast instant coffee adulterated with Robusta.")
          )
        ),
        
        tabPanel(
          "📊 Overview",
          br(),
          div(
            class = "main-panel-card",
            h4("Uploaded data preview"),
            tags$small("First five rows of the uploaded file. Only numeric columns are used as spectral variables."),
            withSpinner(DTOutput("preview_data"))
          ),
          div(
            class = "main-panel-card",
            h4("Prediction summary"),
            tags$small("Counts of samples per predicted class."),
            withSpinner(DTOutput("results_summary")),
            br(),
            withSpinner(plotOutput("bar_distribution", height = "360px")),
            br(),
            downloadButton("download_bar_plot", "Download prediction distribution plot")
          )
        ),
        
        tabPanel(
          "📡 Spectra",
          br(),
          div(
            class = "main-panel-card",
            h4("Spectral profiles"),
            tags$small("Visualise sample spectra as raw or after SNV + Savitzky–Golay 2nd derivative preprocessing."),
            radioButtons(
              "spectra_view",
              "Spectra shown as:",
              choices = c(
                "Raw (unpreprocessed)"        = "raw",
                "SNV + SG 2nd derivative"     = "snv_sg2"
              ),
              selected = "raw",
              inline = TRUE
            ),
            withSpinner(plotOutput("spectra_plot", height = "360px"))
          )
        ),
        
        tabPanel(
          "📈 PCA scores & loadings",
          br(),
          div(
            class = "main-panel-card",
            h4("PCA score plot"),
            tags$small("PCA is applied to SNV + SG 2nd derivative spectra. Both k-NN and RF use these PCA scores as input."),
            withSpinner(plotOutput("pca_dynamic_plot", height = "360px")),
            br(),
            downloadButton("download_pca_scores", "Download PCA score plot")
          ),
          br(),
          div(
            class = "main-panel-card",
            h4("PCA loading plot"),
            tags$small("Select a principal component to inspect its wavelength loadings."),
            selectInput(
              "loading_pc",
              "Component for loading plot:",
              choices = c("PC1" = 1, "PC2" = 2, "PC3" = 3),
              selected = 1,
              width = "200px"
            ),
            br(),
            withSpinner(plotOutput("pca_loading_plot", height = "360px")),
            br(),
            downloadButton("download_pca_loadings", "Download PCA loading plot")
          )
        ),
        
        tabPanel(
          "📐 Cross-validation",
          br(),
          div(
            class = "main-panel-card",
            h4("Cross-validation (training results)"),
            tags$small("k-NN: accuracy versus k. Random Forest: accuracy versus mtry. The optimal settings are marked in red."),
            br(),
            withSpinner(
              fluidRow(
                column(
                  width = 6,
                  plotOutput("cv_rf_plot", height = "320px")
                ),
                column(
                  width = 6,
                  plotOutput("cv_knn_plot", height = "320px")
                )
              )
            )
          )
        ),
        
        tabPanel(
          "⭐ Variable importance",
          br(),
          div(
            class = "main-panel-card",
            h4("Top 20 most important variables"),
            tags$small("Variable importance is computed for the selected classifier (RF or k-NN). Where possible, variables are linked to wavelengths."),
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
            tags$small("Predicted class labels for each sample. Sample_ID is taken from the input file if available, otherwise generated automatically."),
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
            p("Download an example spectral test set derived from real instant coffee samples (test_samples_instant_coffee.xlsx). The TRUE_CLASS column has been removed."),
            downloadButton("download_sample", "Download example CSV"),
            br(), br(),
            h4("Preview of example dataset"),
            withSpinner(DTOutput("sample_preview"))
          )
        ),
        
        tabPanel(
          "⚙️ Preprocessing info",
          br(),
          div(
            class = "main-panel-card",
            h3("Preprocessing pipeline"),
            p("SpectraVision applies the same preprocessing used in the instant coffee case study:"),
            tags$ul(
              tags$li("Standard Normal Variate (SNV): Standardises each spectrum to reduce scatter effects."),
              tags$li("Savitzky–Golay derivative: Smoothing and second-order derivative (window width 13, polynomial order 2, derivative order 2) to enhance subtle spectral features."),
              tags$li("PCA is then applied to the preprocessed spectra, and the resulting scores are used by the k-NN and Random Forest classifiers.")
            )
          )
        ),
        
        tabPanel(
          "✉️ Feedback",
          br(),
          div(
            class = "main-panel-card",
            h3("Feedback"),
            p("Your feedback helps improve this tool for both research and practical use."),
            textAreaInput(
              "user_feedback",
              "Share comments, suggestions, or issues:",
              "",
              width = "100%",
              height = "120px"
            ),
            actionButton("send_feedback", "Submit feedback", class = "btn btn-primary")
          )
        ),
        
        tabPanel(
          "🔍 Diagnostics",
          br(),
          div(
            class = "main-panel-card",
            h3("Diagnostics"),
            p("Basic checks on the uploaded file and app state, useful during development or debugging."),
            verbatimTextOutput("diagnostics")
          )
        ),
        
        tabPanel(
          "ℹ️ About",
          br(),
          div(
            class = "main-panel-card",
            h3("About SpectraVision"),
            p("SpectraVision is an interactive Shiny application that demonstrates how near infrared hyperspectral imaging (900–1700 nm), chemometrics, and machine learning can be used to classify instant Arabica coffee and assess Robusta adulteration patterns."),
            tags$ul(
              tags$li("Models: k-NN and Random Forest trained on PCA scores of NIR-HSI data."),
              tags$li("Preprocessing: SNV combined with Savitzky–Golay second-order derivatives."),
              tags$li("Outputs: Class predictions, prediction summaries, PCA score and loading plots, cross-validation curves, raw vs preprocessed spectra, and variable importance visualisations.")
            ),
            p("Optimised for NIR-HSI spectra (900–1700 nm). Research prototype for demonstration and teaching, not validated for routine quality control.")
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
            p("LinkedIn: ",
              a(href = "https://www.linkedin.com/in/derick-malavi-64742643/",
                "linkedin.com/in/derick-malavi"))
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
  spectra_raw     <- reactiveVal()
  pca_scores      <- reactiveVal()
  pca_predictions <- reactiveVal()
  
  # ---- dynamic theme ----
  observe({
    req(input$theme_choice)
    if (input$theme_choice == "default") {
      session$setCurrentTheme(bs_theme(version = 5))
    } else {
      session$setCurrentTheme(bs_theme(bootswatch = input$theme_choice))
    }
  })
  
  # ---- dynamic PCA choices based on PCA_object ----
  observe({
    if (!is.null(PCA_object) && !is.null(PCA_object$rotation)) {
      ncomp <- ncol(PCA_object$rotation)
      if (ncomp >= 1) {
        pc_choices <- setNames(seq_len(ncomp), paste0("PC", seq_len(ncomp)))
        updateSelectInput(session, "pca_x",
                          choices  = pc_choices,
                          selected = min(1, ncomp))
        updateSelectInput(session, "pca_y",
                          choices  = pc_choices,
                          selected = min(2, ncomp))
        updateSelectInput(session, "loading_pc",
                          choices  = pc_choices,
                          selected = min(1, ncomp))
      }
    }
  })
  
  # ---- run prediction ----
  observeEvent(input$predict_btn, {
    req(input$test_file)
    
    ext <- tools::file_ext(input$test_file$name)
    new_data <- tryCatch(
      {
        switch(
          ext,
          csv  = read_csv(input$test_file$datapath, show_col_types = FALSE),
          txt  = read_delim(input$test_file$datapath, delim = "\t", show_col_types = FALSE),
          tsv  = read_tsv(input$test_file$datapath, show_col_types = FALSE),
          xls  = read_excel(input$test_file$datapath),
          xlsx = read_excel(input$test_file$datapath),
          stop("Unsupported file type.")
        )
      },
      error = function(e) {
        showNotification(paste("Failed to read file:", e$message), type = "error")
        return(NULL)
      }
    )
    req(new_data)
    
    # Preview
    input_preview(head(new_data, 5))
    output$preview_data <- renderDT({
      datatable(input_preview(), options = list(scrollX = TRUE))
    })
    
    # Sample IDs (if present)
    sample_id_col <- NULL
    if ("Sample_ID" %in% names(new_data)) {
      sample_id_col <- "Sample_ID"
    }
    
    # Use only numeric columns as spectra
    numeric_cols <- sapply(new_data, is.numeric)
    spectra_df   <- new_data[, numeric_cols, drop = FALSE]
    
    # Align column names with X_column_names
    if (ncol(spectra_df) == length(X_column_names)) {
      colnames(spectra_df) <- X_column_names
    } else {
      missing_cols <- setdiff(X_column_names, colnames(spectra_df))
      if (length(missing_cols) > 0) {
        showNotification(
          paste(
            "Uploaded data are missing",
            length(missing_cols),
            "required spectral variables. Example:",
            paste(head(missing_cols, 5), collapse = ", "),
            "..."
          ),
          type = "error",
          duration = 8
        )
        return(NULL)
      }
      spectra_df <- spectra_df[, X_column_names, drop = FALSE]
    }
    
    # Store raw spectra for visualisation
    spectra_raw(spectra_df)
    
    # Preprocess (SNV + SG 2nd derivative)
    processed_df <- apply_snv_sg2(spectra_df)
    
    # PCA scores
    X_pca <- predict(PCA_object, processed_df)
    X_pca <- as.data.frame(X_pca)
    colnames(X_pca) <- paste0("PC", seq_len(ncol(X_pca)))
    pca_scores(X_pca)
    
    # Select model
    model <- if (input$model_choice == "rf") rf_model else knn_model
    
    # Predictions (on PCA scores)
    preds_class <- predict(model, newdata = X_pca)
    
    # Sample_ID in output
    if (!is.null(sample_id_col)) {
      sample_ids <- as.character(new_data[[sample_id_col]])
    } else {
      sample_ids <- paste0("Sample", seq_len(nrow(new_data)))
    }
    
    result_df <- data.frame(
      Sample_ID  = sample_ids,
      Prediction = preds_class,
      stringsAsFactors = FALSE
    )
    
    pred_results(result_df)
    pca_predictions(preds_class)
    
    # Outputs
    output$prediction_table <- renderDT({
      datatable(result_df, options = list(scrollX = TRUE))
    })
    
    output$class_summary <- renderPrint({
      table(result_df$Prediction)
    })
    
    output$results_summary <- renderDT({
      as.data.frame(table(result_df$Prediction)) |>
        dplyr::rename(Class = Var1, Count = Freq)
    })
    
    output$bar_distribution <- renderPlot({
      df <- result_df
      ggplot(df, aes(x = Prediction, fill = Prediction)) +
        geom_bar(width = 0.7, color = "black") +
        scale_fill_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = "Distribution of predicted coffee classes",
          x = "Predicted class",
          y = "Number of samples"
        )
    })
    
    # PCA score plot
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
        scale_color_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA score plot:", x_pc, "vs", y_pc),
          x = x_pc,
          y = y_pc,
          color = "Prediction"
        )
    })
    
    showNotification("Prediction complete.", type = "message")
  })
  
  # ========= PCA loadings data =========
  pca_loadings_data <- reactive({
    req(PCA_object)
    validate(need(!is.null(PCA_object$rotation), "No PCA loadings found in PCA object."))
    
    L <- PCA_object$rotation
    df <- data.frame(Variable = rownames(L), L, check.names = FALSE)
    
    # Try to parse wavelength from variable name (e.g. X935 -> 935)
    wl <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", df$Variable)))
    if (!all(is.na(wl))) {
      df$Wavelength_nm <- wl
    } else {
      df$Wavelength_nm <- NA_real_
    }
    df
  })
  
  # ========= PCA loading plot with max points highlighted =========
  output$pca_loading_plot <- renderPlot({
    df <- pca_loadings_data()
    req(nrow(df) > 0)
    
    comp_name <- paste0("PC", input$loading_pc)
    validate(need(comp_name %in% colnames(df), "Selected component not available in PCA loadings."))
    
    df_plot <- df
    df_plot$y_val <- df_plot[[comp_name]]
    
    # Top |loading| points to highlight
    df_plot$abs_loading <- abs(df_plot$y_val)
    top_n <- 5
    df_top <- df_plot[order(-df_plot$abs_loading), ][seq_len(min(top_n, nrow(df_plot))), ]
    
    if (!all(is.na(df_plot$Wavelength_nm))) {
      ggplot(df_plot, aes(x = Wavelength_nm, y = y_val)) +
        geom_line(linewidth = 0.9) +
        geom_point(
          data = df_top,
          aes(x = Wavelength_nm, y = y_val),
          color = "red",
          size  = 2.8
        ) +
        geom_text(
          data = df_top,
          aes(x = Wavelength_nm, y = y_val, label = round(Wavelength_nm)),
          vjust = -1,
          size  = 3,
          color = "red"
        ) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA loading plot for", comp_name),
          x = "Wavelength (nm)",
          y = "Loading"
        )
    } else {
      df_plot$Index <- seq_len(nrow(df_plot))
      df_top$Index  <- seq_len(nrow(df_top))
      
      ggplot(df_plot, aes(x = Index, y = y_val)) +
        geom_line(linewidth = 0.9) +
        geom_point(
          data = df_top,
          aes(x = Index, y = y_val),
          color = "red",
          size  = 2.8
        ) +
        geom_text(
          data = df_top,
          aes(x = Index, y = y_val, label = Index),
          vjust = -1,
          size  = 3,
          color = "red"
        ) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA loading plot for", comp_name),
          x = "Variable index",
          y = "Loading"
        )
    }
  })
  
  output$download_pca_loadings <- downloadHandler(
    filename = function() {
      paste0("spectravision_pca_loadings_PC", input$loading_pc, "_", Sys.Date(), ".png")
    },
    content = function(file) {
      df <- pca_loadings_data()
      req(nrow(df) > 0)
      
      comp_name <- paste0("PC", input$loading_pc)
      validate(need(comp_name %in% colnames(df), "Selected component not available in PCA loadings."))
      
      df_plot <- df
      df_plot$y_val <- df_plot[[comp_name]]
      
      # Top |loading| points to highlight
      df_plot$abs_loading <- abs(df_plot$y_val)
      top_n <- 5
      df_top <- df_plot[order(-df_plot$abs_loading), ][seq_len(min(top_n, nrow(df_plot))), ]
      
      if (!all(is.na(df_plot$Wavelength_nm))) {
        p <- ggplot(df_plot, aes(x = Wavelength_nm, y = y_val)) +
          geom_line(linewidth = 0.9) +
          geom_point(
            data = df_top,
            aes(x = Wavelength_nm, y = y_val),
            color = "red",
            size  = 2.8
          ) +
          geom_text(
            data = df_top,
            aes(x = Wavelength_nm, y = y_val, label = round(Wavelength_nm)),
            vjust = -1,
            size  = 3,
            color = "red"
          ) +
          theme_bw(base_size = 14) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
            axis.title = element_text(color = "black"),
            axis.text  = element_text(color = "black")
          ) +
          labs(
            title = paste("PCA loading plot for", comp_name),
            x = "Wavelength (nm)",
            y = "Loading"
          )
      } else {
        df_plot$Index <- seq_len(nrow(df_plot))
        df_top$Index  <- seq_len(nrow(df_top))
        
        p <- ggplot(df_plot, aes(x = Index, y = y_val)) +
          geom_line(linewidth = 0.9) +
          geom_point(
            data = df_top,
            aes(x = Index, y = y_val),
            color = "red",
            size  = 2.8
          ) +
          geom_text(
            data = df_top,
            aes(x = Index, y = y_val, label = Index),
            vjust = -1,
            size  = 3,
            color = "red"
          ) +
          theme_bw(base_size = 14) +
          theme(
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
            axis.title = element_text(color = "black"),
            axis.text  = element_text(color = "black")
          ) +
          labs(
            title = paste("PCA loading plot for", comp_name),
            x = "Variable index",
            y = "Loading"
          )
      }
      ggsave(file, p, width = 7, height = 5, dpi = 300)
    }
  )
  
  # ========= Spectra plot (raw vs SNV+SG2) =========
  output$spectra_plot <- renderPlot({
    req(spectra_raw())
    mat <- spectra_raw()
    
    # Apply view-specific preprocessing for plotting only
    if (input$spectra_view == "snv_sg2") {
      mat <- as.matrix(apply_snv_sg2(mat))
    } else {
      mat <- as.matrix(mat)  # raw
    }
    
    # Limit number of spectra to avoid clutter
    n_samp <- min(30, nrow(mat))
    mat <- mat[seq_len(n_samp), , drop = FALSE]
    
    df_long <- as.data.frame(mat)
    df_long$Sample <- factor(paste0("S", seq_len(nrow(df_long))))
    
    df_long <- df_long %>%
      tidyr::pivot_longer(
        cols = -Sample,
        names_to = "Var",
        values_to = "Intensity"
      )
    
    # Map variable names to numeric wavelengths if possible
    wl_num <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", colnames(mat))))
    if (length(wl_num) == ncol(mat) && !any(is.na(wl_num))) {
      wl_map <- data.frame(Var = colnames(mat), Wavelength = wl_num)
      df_long <- df_long %>%
        dplyr::left_join(wl_map, by = "Var")
      x_lab <- "Wavelength (nm)"
      
      ggplot(df_long, aes(x = Wavelength, y = Intensity, group = Sample, color = Sample)) +
        geom_line(alpha = 0.6) +
        scale_color_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black"),
          legend.position = "none"
        ) +
        labs(
          title = "Sample spectra",
          x = x_lab,
          y = "Intensity"
        )
    } else {
      df_long$WavelengthIndex <- as.numeric(gsub("X", "", df_long$Var))
      ggplot(df_long, aes(x = WavelengthIndex, y = Intensity, group = Sample, color = Sample)) +
        geom_line(alpha = 0.6) +
        scale_color_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black"),
          legend.position = "none"
        ) +
        labs(
          title = "Sample spectra",
          x = "Wavelength index",
          y = "Intensity"
        )
    }
  })
  
  # ========= Cross-validation plots for RF and k-NN =========
  output$cv_rf_plot <- renderPlot({
    df <- rf_model$results
    best_mtry <- rf_model$bestTune$mtry
    best_acc  <- df$Accuracy[df$mtry == best_mtry]
    
    ggplot(df, aes(x = mtry, y = Accuracy)) +
      geom_line(color = "#4f46e5", linewidth = 1) +
      geom_point(color = "#111827", size = 2.5) +
      annotate(
        "segment",
        x = best_mtry, xend = best_mtry,
        y = min(df$Accuracy, na.rm = TRUE), yend = best_acc,
        linetype = "dotted", color = "red", linewidth = 0.8
      ) +
      annotate(
        "point",
        x = best_mtry, y = best_acc,
        color = "red", size = 3
      ) +
      annotate(
        "text",
        x = best_mtry, y = best_acc,
        label = paste0("Best mtry = ", best_mtry),
        vjust = -1,
        color = "red",
        size = 3.5
      ) +
      theme_bw(base_size = 13) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title = element_text(color = "black"),
        axis.text  = element_text(color = "black")
      ) +
      labs(
        title = "Random Forest CV accuracy by mtry",
        x = "mtry",
        y = "Accuracy"
      )
  })
  
  output$cv_knn_plot <- renderPlot({
    df <- knn_model$results
    best_k   <- knn_model$bestTune$k
    best_acc <- df$Accuracy[df$k == best_k]
    
    ggplot(df, aes(x = k, y = Accuracy)) +
      geom_line(color = "#16a34a", linewidth = 1) +
      geom_point(color = "#111827", size = 2.5) +
      annotate(
        "segment",
        x = best_k, xend = best_k,
        y = min(df$Accuracy, na.rm = TRUE), yend = best_acc,
        linetype = "dotted", color = "red", linewidth = 0.8
      ) +
      annotate(
        "point",
        x = best_k, y = best_acc,
        color = "red", size = 3
      ) +
      annotate(
        "text",
        x = best_k, y = best_acc,
        label = paste0("Best k = ", best_k),
        vjust = -1,
        color = "red",
        size = 3.5
      ) +
      theme_bw(base_size = 13) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title = element_text(color = "black"),
        axis.text  = element_text(color = "black")
      ) +
      labs(
        title = "k-NN CV accuracy by k",
        x = "k",
        y = "Accuracy"
      )
  })
  
  # ========= Variable importance =========
  varimp_data <- reactive({
    req(input$model_choice)
    model <- if (input$model_choice == "rf") rf_model else knn_model
    
    imp <- tryCatch({
      varImp(model, scale = TRUE)
    }, error = function(e) NULL)
    
    if (is.null(imp) || is.null(imp$importance)) {
      return(data.frame())
    }
    
    df <- as.data.frame(imp$importance)
    df$Variable <- rownames(df)
    
    if ("Overall" %in% names(df)) {
      df$Importance <- df$Overall
    } else {
      score_cols <- setdiff(names(df), "Variable")
      df$Importance <- apply(df[, score_cols, drop = FALSE], 1, mean, na.rm = TRUE)
    }
    
    # Try to parse wavelength from variable name
    wl <- suppressWarnings(as.numeric(gsub("[^0-9.]", "", df$Variable)))
    if (length(wl) == nrow(df) && !all(is.na(wl))) {
      df$Wavelength_nm <- wl
    } else {
      df$Wavelength_nm <- NA_real_
    }
    
    df <- df[order(-abs(df$Importance)), ]
    head(df, 20)
  })
  
  output$varimp_plot <- renderPlot({
    df <- varimp_data()
    req(nrow(df) > 0)
    
    ggplot(df, aes(x = reorder(Variable, Importance),
                   y = Importance,
                   fill = Importance)) +
      geom_col(width = 0.8, color = "black") +
      scale_fill_viridis_c(option = "C", begin = 0, end = 1) +
      coord_flip() +
      theme_bw(base_size = 13) +
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
        axis.title = element_text(color = "black"),
        axis.text  = element_text(color = "black")
      ) +
      labs(
        x = "Variable",
        y = "Importance",
        title = "Top 20 important variables"
      )
  })
  
  output$varimp_table <- renderDT({
    df <- varimp_data()
    if (nrow(df) == 0) {
      return(datatable(
        data.frame(Message = "Variable importance not available for this model."),
        options = list(dom = "t"),
        rownames = FALSE
      ))
    }
    cols <- c("Variable", "Wavelength_nm", "Importance")
    datatable(df[, cols],
              options = list(pageLength = 20, scrollX = TRUE),
              rownames = FALSE)
  })
  
  output$download_varimp <- downloadHandler(
    filename = function() {
      paste0("spectravision_varimp_", input$model_choice, "_", Sys.Date(), ".png")
    },
    content = function(file) {
      df <- varimp_data()
      req(nrow(df) > 0)
      
      p <- ggplot(df, aes(x = reorder(Variable, Importance),
                          y = Importance,
                          fill = Importance)) +
        geom_col(width = 0.8, color = "black") +
        scale_fill_viridis_c(option = "C", begin = 0, end = 1) +
        coord_flip() +
        theme_bw(base_size = 13) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          x = "Variable",
          y = "Importance",
          title = "Top 20 important variables"
        )
      ggsave(file, p, width = 7, height = 5, dpi = 300)
    }
  )
  
  # ---- diagnostics ----
  output$diagnostics <- renderPrint({
    tryCatch(
      {
        list(
          "Columns in uploaded file"              = names(input_preview()),
          "First few expected spectral variables" = head(X_column_names),
          "Number of preview rows"                = if (is.null(input_preview())) 0 else nrow(input_preview()),
          "Spectra loaded (rows)"                 = if (is.null(spectra_raw())) 0 else nrow(spectra_raw())
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
  
  # ---- download bar plot ----
  output$download_bar_plot <- downloadHandler(
    filename = function() {
      paste0("spectravision_prediction_distribution_", Sys.Date(), ".png")
    },
    content = function(file) {
      req(pred_results())
      df <- pred_results()
      
      p <- ggplot(df, aes(x = Prediction, fill = Prediction)) +
        geom_bar(width = 0.7, color = "black") +
        scale_fill_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = "Distribution of predicted coffee classes",
          x = "Predicted class",
          y = "Number of samples"
        )
      ggsave(file, p, width = 7, height = 5, dpi = 300)
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
        scale_color_viridis_d(option = "C", begin = 0, end = 1) +
        theme_bw(base_size = 14) +
        theme(
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold", color = "black"),
          axis.title = element_text(color = "black"),
          axis.text  = element_text(color = "black")
        ) +
        labs(
          title = paste("PCA score plot:", x_pc, "vs", y_pc),
          x = x_pc,
          y = y_pc,
          color = "Prediction"
        )
      ggsave(file, p, width = 7, height = 5, dpi = 300)
    }
  )
  
  # ---- sample data (real test file, cleaned) ----
  output$download_sample <- downloadHandler(
    filename = function() { "Instant_Coffee_Test_Samples.csv" },
    content = function(file) {
      # Read the real instant coffee test samples from Excel
      sample_data <- readxl::read_excel("test_samples_instant_coffee.xlsx")
      
      # Remove TRUE_CLASS column if it exists
      if ("TRUE_CLASS" %in% names(sample_data)) {
        sample_data <- dplyr::select(sample_data, -TRUE_CLASS)
      }
      
      # Ensure there is a Sample_ID column and make it character
      if ("Sample_ID" %in% names(sample_data)) {
        sample_data$Sample_ID <- as.character(sample_data$Sample_ID)
      } else {
        id_col <- names(sample_data)[1]
        names(sample_data)[1] <- "Sample_ID"
        sample_data$Sample_ID <- as.character(sample_data$Sample_ID)
      }
      
      write.csv(sample_data, file, row.names = FALSE)
    },
    contentType = "text/csv"
  )
  
  output$sample_preview <- renderDT({
    sample_data <- readxl::read_excel("test_samples_instant_coffee.xlsx")
    
    if ("TRUE_CLASS" %in% names(sample_data)) {
      sample_data <- dplyr::select(sample_data, -TRUE_CLASS)
    }
    
    if ("Sample_ID" %in% names(sample_data)) {
      sample_data$Sample_ID <- as.character(sample_data$Sample_ID)
    } else {
      id_col <- names(sample_data)[1]
      names(sample_data)[1] <- "Sample_ID"
      sample_data$Sample_ID <- as.character(sample_data$Sample_ID)
    }
    
    datatable(head(sample_data, 10), options = list(scrollX = TRUE))
  })
  
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
        file  = feedback_file,
        append = TRUE,
        sep    = ",",
        row.names = FALSE,
        col.names = FALSE
      )
    } else {
      write.table(
        feedback,
        file  = feedback_file,
        append = FALSE,
        sep    = ",",
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
    spectra_raw(NULL)
    pca_scores(NULL)
    pca_predictions(NULL)
    
    output$prediction_table   <- renderDT(NULL)
    output$class_summary      <- renderPrint(NULL)
    output$results_summary    <- renderDT(NULL)
    output$bar_distribution   <- renderPlot(NULL)
    output$pca_dynamic_plot   <- renderPlot(NULL)
    output$pca_loading_plot   <- renderPlot(NULL)
    output$preview_data       <- renderDT(NULL)
    output$varimp_plot        <- renderPlot(NULL)
    output$varimp_table       <- renderDT(NULL)
    output$spectra_plot       <- renderPlot(NULL)
    
    updateFileInput(session, "test_file", NULL)
    updateTextInput(session, "user_feedback", value = "")
  })
}

shinyApp(ui = ui, server = server)
