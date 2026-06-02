library(shiny)
library(DT)
library(dplyr)
library(readr)
library(ggplot2)
library(naniar)
library(scales)

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH <- normalizePath("../data/c15_enriched.csv")

# Columns shown in the editor (subset of 285 total)
KEY_COLS <- c(
  "病歷號(2)", "sex", "age", "bmi",
  "subsite", "histology_group", "grade_label",
  "clin_stage", "path_stage", "stage", "stage_group",
  "surgery", "radiation", "chemo",
  "alcohol", "smoker", "betel_nut",
  "vital_status", "os_days"
)

# Column display labels (English short names for the table header)
COL_LABELS <- c(
  "ID", "Sex", "Age", "BMI",
  "Subsite", "Histology", "Grade",
  "cStage", "pStage", "Stage", "Stage Grp",
  "Surgery", "RT", "Chemo",
  "Alcohol", "Smoker", "Betel Nut",
  "Status", "OS (days)"
)

# Allowed values per column (NULL = free text / numeric)
VALID_VALS <- list(
  sex          = c("Male", "Female"),
  vital_status = c("Alive", "Dead"),
  surgery      = c("TRUE", "FALSE"),
  radiation    = c("TRUE", "FALSE"),
  chemo        = c("TRUE", "FALSE"),
  smoker       = c("Yes", "No"),
  betel_nut    = c("Yes", "No"),
  grade_label  = c("Well differentiated (G1)", "Moderately differentiated (G2)",
                   "Poorly differentiated (G3)", "Undifferentiated (G4)", "Unknown/NA"),
  stage_group  = c("I", "II", "III", "IV"),
  clin_stage   = c("0","1","1A","1B","2","2A","2B","2E",
                   "3","3A","3B","3C","4","4A","4B"),
  path_stage   = c("0","1","1A","1B","2","2A","2B",
                   "3","3A","3B","3C","4","4A","4B"),
  stage        = c("0","1","1A","1B","2","2A","2B","2E",
                   "3","3A","3B","3C","4","4A","4B"),
  alcohol      = c("0","1","2","3","4")
)

ALCOHOL_LABELS <- c(
  "0" = "0 – None",
  "1" = "1 – Social",
  "2" = "2 – Regular",
  "3" = "3 – Heavy",
  "4" = "4 – Former"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

load_data <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  # Normalise boolean columns to character for display
  for (col in c("surgery", "radiation", "chemo", "immunotherapy", "targeted")) {
    if (col %in% names(df)) df[[col]] <- as.character(df[[col]])
  }
  df
}

# Return % complete for a vector
pct_complete <- function(x) round(100 * mean(!is.na(x)), 1)

validate_cell <- function(col, value) {
  if (is.na(value) || value == "" || value == "NA") return(TRUE)  # NA always OK
  if (!col %in% names(VALID_VALS)) return(TRUE)                   # no rule
  value %in% VALID_VALS[[col]]
}

# ── UI ───────────────────────────────────────────────────────────────────────

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { font-family: 'Segoe UI', sans-serif; font-size: 13px; }
    .nav-tabs > li > a { font-size: 13px; }
    .miss-cell { background-color: #fee2e2 !important; color: #991b1b !important; }
    .warn-cell { background-color: #fef9c3 !important; color: #713f12 !important; }
    .ok-cell   { background-color: #dcfce7 !important; color: #166534 !important; }
    .badge-miss { background:#ef4444; color:#fff; padding:2px 7px; border-radius:10px; font-size:11px; }
    .badge-ok   { background:#22c55e; color:#fff; padding:2px 7px; border-radius:10px; font-size:11px; }
    .changelog-tbl td { font-size:12px; }
    h4 { color: #1e40af; margin-top:18px; }
  "))),

  titlePanel("C15 食道癌登錄資料品質編輯器"),

  sidebarLayout(
    sidebarPanel(
      width = 2,

      h5("資料來源"),
      fileInput("upload", "上傳 CSV（選用）", accept = ".csv"),
      actionButton("reload", "重新載入預設", class = "btn-sm btn-secondary",
                   style = "width:100%; margin-top:4px"),
      hr(),

      h5("篩選列"),
      selectInput("filter_col", "依欄位篩選", choices = c("(全部)", KEY_COLS)),
      selectInput("filter_val", "值", choices = c("(全部)", "NA / 缺失"), multiple = FALSE),
      hr(),

      h5("篩選欄"),
      checkboxGroupInput("show_cols", "顯示欄位",
                         choices  = setNames(KEY_COLS, COL_LABELS),
                         selected = KEY_COLS),
      hr(),
      tags$small("版本 1.0 · C15 Esophageal")
    ),

    mainPanel(
      width = 10,
      tabsetPanel(id = "tabs",

        # ── Tab 1: Missing overview ──────────────────────────────────────────
        tabPanel("缺失概覽",
          fluidRow(
            column(12,
              h4("關鍵分析欄位完整度"),
              plotOutput("miss_bar", height = "320px"),
              hr(),
              h4("缺失模式熱圖"),
              plotOutput("miss_map", height = "380px")
            )
          )
        ),

        # ── Tab 2: Data editor ───────────────────────────────────────────────
        tabPanel("資料編輯",
          fluidRow(
            column(12,
              div(style = "margin: 8px 0;",
                uiOutput("miss_summary_badge"),
                span(style = "margin-left:16px; color:#6b7280; font-size:12px;",
                     "點擊任意儲存格直接編輯；NA 代表缺失值（紅色背景）")
              ),
              DTOutput("editor_tbl"),
              uiOutput("validation_msg")
            )
          )
        ),

        # ── Tab 3: Changelog & Export ────────────────────────────────────────
        tabPanel("變更紀錄 & 匯出",
          fluidRow(
            column(6,
              h4("變更紀錄"),
              DTOutput("changelog_tbl")
            ),
            column(6,
              h4("匯出"),
              p("匯出包含全部 285 個欄位，已修改值將更新至原始欄位。"),
              downloadButton("dl_edited",   "下載已修改資料 (CSV)", class = "btn-primary"),
              br(), br(),
              downloadButton("dl_changelog", "下載變更紀錄 (CSV)", class = "btn-secondary"),
              hr(),
              h4("修改統計"),
              verbatimTextOutput("edit_stats")
            )
          )
        )
      )
    )
  )
)

# ── Server ───────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # Reactive: full dataset (all 285 cols)
  rv <- reactiveValues(
    full  = NULL,   # complete original data
    data  = NULL,   # current (edited) data — KEY_COLS only
    log   = data.frame(
      Time     = character(),
      Row      = integer(),
      ID       = character(),
      Column   = character(),
      OldValue = character(),
      NewValue = character(),
      Valid    = character(),
      stringsAsFactors = FALSE
    )
  )

  # Load default on startup
  observe({
    req(is.null(rv$full))
    full <- load_data(DATA_PATH)
    rv$full <- full
    rv$data <- full[, intersect(KEY_COLS, names(full)), drop = FALSE]
  })

  # Reload from file upload
  observeEvent(input$upload, {
    req(input$upload)
    full <- load_data(input$upload$datapath)
    rv$full <- full
    rv$data <- full[, intersect(KEY_COLS, names(full)), drop = FALSE]
    rv$log   <- rv$log[0, ]
    showNotification("已載入上傳的 CSV", type = "message")
  })

  # Reload default
  observeEvent(input$reload, {
    full <- load_data(DATA_PATH)
    rv$full <- full
    rv$data <- full[, intersect(KEY_COLS, names(full)), drop = FALSE]
    rv$log   <- rv$log[0, ]
    showNotification("已重新載入預設資料", type = "message")
  })

  # ── Filter sidebar ────────────────────────────────────────────────────────

  observeEvent(input$filter_col, {
    req(rv$data)
    col <- input$filter_col
    if (col == "(全部)" || !col %in% names(rv$data)) {
      updateSelectInput(session, "filter_val", choices = c("(全部)", "NA / 缺失"))
    } else {
      vals <- sort(unique(as.character(rv$data[[col]])))
      choices <- c("(全部)", "NA / 缺失", vals[vals != "NA"])
      updateSelectInput(session, "filter_val", choices = choices)
    }
  })

  # Filtered view indices
  filtered_rows <- reactive({
    req(rv$data)
    rows <- seq_len(nrow(rv$data))
    col <- input$filter_col
    val <- input$filter_val
    if (!is.null(col) && col != "(全部)" && col %in% names(rv$data)) {
      if (!is.null(val) && val == "NA / 缺失") {
        rows <- which(is.na(rv$data[[col]]))
      } else if (!is.null(val) && val != "(全部)") {
        rows <- which(as.character(rv$data[[col]]) == val)
      }
    }
    rows
  })

  # Visible columns
  show_cols <- reactive({
    req(input$show_cols)
    intersect(input$show_cols, names(rv$data))
  })

  # ── Missing overview ──────────────────────────────────────────────────────

  output$miss_bar <- renderPlot({
    req(rv$data)
    df <- rv$data
    pcts <- sapply(names(df), function(c) pct_complete(df[[c]]))
    tbl <- data.frame(
      col  = factor(names(pcts), levels = names(pcts)[order(pcts)]),
      pct  = pcts,
      fill = ifelse(pcts >= 90, "#22c55e",
             ifelse(pcts >= 70, "#f59e0b", "#ef4444"))
    )
    # Use COL_LABELS where available
    label_map <- setNames(COL_LABELS, KEY_COLS)
    tbl$label <- ifelse(tbl$col %in% names(label_map),
                        label_map[as.character(tbl$col)],
                        as.character(tbl$col))
    tbl$col <- factor(tbl$label, levels = tbl$label[order(pcts)])

    ggplot(tbl, aes(x = col, y = pct, fill = fill)) +
      geom_col(width = 0.7) +
      geom_text(aes(label = paste0(pct, "%")), hjust = -0.1, size = 3.2) +
      geom_hline(yintercept = c(70, 90), linetype = "dashed",
                 colour = c("#f59e0b", "#22c55e"), linewidth = 0.5) +
      scale_fill_identity() +
      scale_y_continuous(limits = c(0, 110), labels = function(x) paste0(x, "%")) +
      coord_flip() +
      labs(x = NULL, y = "完整率 (%)", title = NULL) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.minor = element_blank(),
            panel.grid.major.y = element_blank())
  })

  output$miss_map <- renderPlot({
    req(rv$data)
    df <- rv$data[filtered_rows(), show_cols(), drop = FALSE]
    if (nrow(df) == 0) return(NULL)
    # Convert all to character then NA for missingness check
    df_char <- as.data.frame(lapply(df, function(x) {
      x_c <- as.character(x)
      x_c[x_c == "NA" | is.na(x)] <- NA_character_
      x_c
    }), stringsAsFactors = FALSE)

    naniar::vis_miss(df_char, warn_large_data = FALSE) +
      labs(title = paste0("缺失模式 (", nrow(df), " 列)")) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9))
  })

  # ── Editor table ──────────────────────────────────────────────────────────

  output$miss_summary_badge <- renderUI({
    req(rv$data)
    df <- rv$data[, show_cols(), drop = FALSE]
    na_count <- sum(sapply(df, function(x) sum(is.na(x))))
    total    <- nrow(df) * ncol(df)
    pct_miss <- round(100 * na_count / total, 1)
    cls  <- if (pct_miss > 20) "badge-miss" else "badge-ok"
    tags$span(
      class = cls,
      paste0("缺失: ", na_count, " 格 (", pct_miss, "%)")
    )
  })

  output$editor_tbl <- renderDT({
    req(rv$data)
    rows <- filtered_rows()
    cols <- show_cols()
    df_show <- rv$data[rows, cols, drop = FALSE]

    # Build column-level format calls
    dt <- datatable(
      df_show,
      rownames   = FALSE,
      selection  = "single",
      editable   = list(target = "cell"),
      filter     = "top",
      extensions = "Scroller",
      options    = list(
        scrollX    = TRUE,
        scrollY    = "480px",
        scroller   = TRUE,
        deferRender = TRUE,
        pageLength = 50,
        dom        = "lfrtip",
        columnDefs = list(list(width = "80px", targets = "_all"))
      )
    )

    # Colour NA cells red; cells with valid-value violations yellow
    for (j in seq_along(cols)) {
      col <- cols[[j]]
      vals <- df_show[[col]]
      na_rows <- which(is.na(vals)) - 1L  # 0-indexed for JS
      dt <- dt %>%
        formatStyle(col,
          backgroundColor = styleEqual(
            c(NA_character_),
            c("#fee2e2")
          )
        )
    }
    dt
  }, server = TRUE)

  # Capture cell edits
  observeEvent(input$editor_tbl_cell_edit, {
    info <- input$editor_tbl_cell_edit
    req(rv$data)

    rows <- filtered_rows()
    cols <- show_cols()

    # Map table row/col back to rv$data
    real_row <- rows[info$row]
    real_col <- cols[info$col + 1L]  # DT col is 0-indexed when rownames=FALSE

    old_val <- as.character(rv$data[[real_col]][real_row])
    new_val <- if (info$value == "" || info$value == "NA") NA_character_
               else as.character(info$value)

    # Validate
    is_valid <- validate_cell(real_col, new_val)

    if (!is_valid) {
      showNotification(
        paste0("無效值「", new_val, "」—— 欄位「", real_col, "」允許: ",
               paste(VALID_VALS[[real_col]], collapse = ", ")),
        type = "error", duration = 6
      )
      # Revert the table cell
      proxy <- dataTableProxy("editor_tbl")
      replaceData(proxy, rv$data[filtered_rows(), show_cols(), drop = FALSE],
                  resetPaging = FALSE, rownames = FALSE)
      return()
    }

    # Apply edit to rv$data and rv$full
    rv$data[[real_col]][real_row] <- new_val
    rv$full[[real_col]][real_row] <- new_val

    # Log
    patient_id_col <- "病歷號(2)"
    pid <- if (patient_id_col %in% names(rv$data)) rv$data[[patient_id_col]][real_row] else real_row

    rv$log <- rbind(rv$log, data.frame(
      Time     = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      Row      = real_row,
      ID       = as.character(pid),
      Column   = real_col,
      OldValue = old_val,
      NewValue = ifelse(is.na(new_val), "NA", new_val),
      Valid    = ifelse(is_valid, "✓", "✗"),
      stringsAsFactors = FALSE
    ))

    showNotification(
      paste0("已更新：", real_col, " [列 ", real_row, "] → ", ifelse(is.na(new_val), "NA", new_val)),
      type = "message", duration = 3
    )
  })

  output$validation_msg <- renderUI({
    if (nrow(rv$log) == 0) return(NULL)
    n_edits <- nrow(rv$log)
    tags$div(
      style = "margin-top:6px; color:#6b7280; font-size:12px;",
      paste0("本次編輯共 ", n_edits, " 筆變更")
    )
  })

  # ── Changelog ────────────────────────────────────────────────────────────

  output$changelog_tbl <- renderDT({
    datatable(rv$log, rownames = FALSE, options = list(pageLength = 15, dom = "tp"),
              class = "changelog-tbl")
  })

  output$edit_stats <- renderText({
    req(rv$data)
    df   <- rv$data
    pcts <- sapply(names(df), pct_complete)
    n_edits <- nrow(rv$log)
    paste0(
      "已編輯 ", n_edits, " 格\n",
      "目前 stage_group 完整率: ",    pct_complete(df$stage_group),    "%\n",
      "目前 clin_stage 完整率:  ",    pct_complete(df$clin_stage),     "%\n",
      "目前 grade_label 完整率: ",    pct_complete(df$grade_label),    "%\n",
      "目前 alcohol 完整率:     ",    pct_complete(df$alcohol),        "%\n",
      "目前 smoker 完整率:      ",    pct_complete(df$smoker),         "%\n",
      "目前 betel_nut 完整率:   ",    pct_complete(df$betel_nut),      "%\n",
      "目前 bmi 完整率:         ",    pct_complete(df$bmi),            "%"
    )
  })

  # ── Downloads ────────────────────────────────────────────────────────────

  output$dl_edited <- downloadHandler(
    filename = function() paste0("c15_edited_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content  = function(file) write_csv(rv$full, file)
  )

  output$dl_changelog <- downloadHandler(
    filename = function() paste0("c15_changelog_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content  = function(file) write_csv(rv$log, file)
  )
}

shinyApp(ui, server)
