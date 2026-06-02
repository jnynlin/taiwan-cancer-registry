library(shiny)
library(DT)
library(dplyr)
library(readr)
library(ggplot2)
library(naniar)

# ── Configuration ─────────────────────────────────────────────────────────────

# Local run: auto-load if file exists; shinyapps.io: upload-only
DATA_PATH <- normalizePath("../data/c15_enriched.csv", mustWork = FALSE)
AUTO_LOAD <- file.exists(DATA_PATH)

KEY_COLS <- c(
  "病歷號(2)", "sex", "age", "bmi",
  "subsite", "histology_group", "grade_label",
  "clin_stage", "path_stage", "stage", "stage_group",
  "surgery", "radiation", "chemo",
  "alcohol", "smoker", "betel_nut",
  "vital_status", "os_days"
)

COL_LABELS <- c(
  "ID", "Sex", "Age", "BMI",
  "Subsite", "Histology", "Grade",
  "cStage", "pStage", "Stage", "Stage Grp",
  "Surgery", "RT", "Chemo",
  "Alcohol", "Smoker", "Betel Nut",
  "Status", "OS (days)"
)

# ── Editable fields (col → input spec) ───────────────────────────────────────
# Read-only in modal: 病歷號(2), subsite, histology_group

EDIT_FIELDS <- list(
  list(col="sex",         id="e_sex",    label="Sex",
       type="select", choices=c("Male","Female")),
  list(col="age",         id="e_age",    label="Age (yr)",
       type="numeric", min=0, max=120, step=1),
  list(col="bmi",         id="e_bmi",    label="BMI (kg/m²)",
       type="numeric", min=10, max=60, step=0.1),
  list(col="grade_label", id="e_grade",  label="Grade",
       type="select",
       choices=c("Well differentiated (G1)","Moderately differentiated (G2)",
                 "Poorly differentiated (G3)","Undifferentiated (G4)","Unknown/NA")),
  list(col="clin_stage",  id="e_cstage", label="Clinical Stage",
       type="select",
       choices=c("0","1","1A","1B","2","2A","2B","2E",
                 "3","3A","3B","3C","4","4A","4B")),
  list(col="path_stage",  id="e_pstage", label="Pathologic Stage",
       type="select",
       choices=c("0","1","1A","1B","2","2A","2B",
                 "3","3A","3B","3C","4","4A","4B")),
  list(col="stage",       id="e_stage",  label="Stage (combined)",
       type="select",
       choices=c("0","1","1A","1B","2","2A","2B","2E",
                 "3","3A","3B","3C","4","4A","4B")),
  list(col="stage_group", id="e_sgrp",   label="Stage Group",
       type="select", choices=c("I","II","III","IV")),
  list(col="surgery",     id="e_surg",   label="Surgery",
       type="select", choices=c("TRUE","FALSE")),
  list(col="radiation",   id="e_rad",    label="Radiation (RT)",
       type="select", choices=c("TRUE","FALSE")),
  list(col="chemo",       id="e_chemo",  label="Chemotherapy",
       type="select", choices=c("TRUE","FALSE")),
  list(col="alcohol",     id="e_alc",    label="Alcohol",
       type="select",
       choices=c("0 – None"="0","1 – Social"="1","2 – Regular"="2",
                 "3 – Heavy"="3","4 – Former"="4")),
  list(col="smoker",      id="e_smk",    label="Smoker",
       type="select", choices=c("Yes","No")),
  list(col="betel_nut",   id="e_betel",  label="Betel Nut",
       type="select", choices=c("Yes","No")),
  list(col="vital_status",id="e_vital",  label="Vital Status",
       type="select", choices=c("Alive","Dead")),
  list(col="os_days",     id="e_os",     label="OS (days)",
       type="numeric", min=0, max=18250, step=1)
)

# ── Helpers ───────────────────────────────────────────────────────────────────

load_data <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  for (col in c("surgery","radiation","chemo","immunotherapy","targeted"))
    if (col %in% names(df)) df[[col]] <- as.character(df[[col]])
  df
}

pct_complete <- function(x) round(100 * mean(!is.na(x)), 1)

NA_SENTINEL <- "__NA__"

build_modal_input <- function(fld, current_val) {
  val <- if (is.na(current_val) || current_val == "NA") NA else current_val
  na_choice <- setNames(NA_SENTINEL, "∅ (缺失/NA)")

  if (fld$type == "select") {
    choices  <- c(na_choice, setNames(fld$choices, fld$choices))
    selected <- if (is.na(val)) NA_SENTINEL else as.character(val)
    selectInput(fld$id, fld$label, choices = choices, selected = selected,
                width = "100%")
  } else {
    num_val <- suppressWarnings(as.numeric(val))
    numericInput(fld$id, fld$label,
                 value = ifelse(is.na(num_val), NA, num_val),
                 min = fld$min, max = fld$max, step = fld$step,
                 width = "100%")
  }
}

# ── CSS ───────────────────────────────────────────────────────────────────────

APP_CSS <- "
body { font-family: 'Segoe UI', sans-serif; font-size: 13px; }
.badge-miss { background:#ef4444; color:#fff; padding:2px 8px; border-radius:10px; font-size:11px; }
.badge-ok   { background:#22c55e; color:#fff; padding:2px 8px; border-radius:10px; font-size:11px; }
.edit-bar   { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;
              padding:8px 14px; margin-bottom:8px; display:flex; align-items:center; gap:12px; }
.row-info   { color:#64748b; font-size:12px; flex:1; }
.modal-section { border-left:3px solid #3b82f6; padding-left:10px; margin-bottom:12px; }
.modal-section-title { font-size:11px; font-weight:600; color:#3b82f6;
                       text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }
.readonly-field { background:#f1f5f9; padding:5px 8px; border-radius:4px;
                  font-size:12px; color:#475569; border:1px solid #e2e8f0; margin-bottom:8px; }
.readonly-label { font-size:11px; color:#94a3b8; margin-bottom:2px; }
table.dataTable tbody tr.selected td { background-color:#dbeafe !important; }
table.dataTable tbody tr:hover td { background-color:#f0f9ff; cursor:pointer; }
.na-cell { color:#dc2626; font-style:italic; }
h4 { color:#1e40af; margin-top:18px; }
.changelog-tbl td { font-size:12px; }
"

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- fluidPage(
  tags$head(tags$style(HTML(APP_CSS))),
  titlePanel("C15 食道癌登錄資料品質編輯器 v2"),

  sidebarLayout(
    sidebarPanel(
      width = 2,
      h5("資料來源"),
      fileInput("upload", "上傳 CSV（選用）", accept = ".csv"),
      actionButton("reload", "重新載入預設", class = "btn-sm btn-secondary",
                   style = "width:100%; margin-top:4px"),
      hr(),
      h5("篩選列"),
      selectInput("filter_col", "依欄位", choices = c("(全部)", KEY_COLS)),
      selectInput("filter_val", "值",     choices = c("(全部)", "NA / 缺失")),
      hr(),
      h5("顯示欄位"),
      checkboxGroupInput("show_cols", NULL,
                         choices  = setNames(KEY_COLS, COL_LABELS),
                         selected = KEY_COLS),
      hr(),
      tags$small("v2.0 · C15 Esophageal")
    ),

    mainPanel(
      width = 10,
      tabsetPanel(id = "tabs",

        # ── Tab 1: Missing overview ──────────────────────────────────────
        tabPanel("缺失概覽",
          fluidRow(column(12,
            h4("關鍵欄位完整度"),
            plotOutput("miss_bar", height = "320px"),
            hr(),
            h4("缺失模式熱圖"),
            plotOutput("miss_map", height = "380px")
          ))
        ),

        # ── Tab 2: Data editor ───────────────────────────────────────────
        tabPanel("資料編輯",
          uiOutput("upload_prompt"),
          uiOutput("edit_bar_ui"),   # server renders when data loaded
          DTOutput("editor_tbl"),    # always in DOM so proxy works
          uiOutput("edit_count_msg")
        ),

        # ── Tab 3: Changelog & Export ────────────────────────────────────
        tabPanel("變更紀錄 & 匯出",
          fluidRow(
            column(7,
              h4("變更紀錄"),
              DTOutput("changelog_tbl")
            ),
            column(5,
              h4("匯出"),
              p(style="font-size:12px;",
                "匯出含全部 285 欄，已修改值已更新。"),
              downloadButton("dl_edited",    "下載修改後資料 (CSV)", class = "btn-primary"),
              br(), br(),
              downloadButton("dl_changelog", "下載變更紀錄 (CSV)", class = "btn-secondary"),
              hr(),
              h4("完整率快覽"),
              verbatimTextOutput("edit_stats")
            )
          )
        )
      )
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  rv <- reactiveValues(
    full     = NULL,
    data     = NULL,
    selected = NULL,   # real row index in rv$data
    log = data.frame(
      Time=character(), Row=integer(), ID=character(),
      Column=character(), OldValue=character(), NewValue=character(),
      stringsAsFactors=FALSE
    )
  )

  output$edit_bar_ui <- renderUI({
    req(rv$full)
    div(class = "edit-bar",
      uiOutput("miss_badge"),
      uiOutput("row_info"),
      actionButton("open_edit", "編輯選取列", icon = icon("pen"),
                   class = "btn-primary btn-sm"),
      actionButton("clear_sel", "取消選取", class = "btn-sm btn-secondary")
    )
  })

  output$upload_prompt <- renderUI({
    if (!is.null(rv$full)) return(NULL)
    div(style = "text-align:center; padding:60px 20px;",
      tags$div(style = "font-size:48px; color:#cbd5e1;", icon("file-csv")),
      tags$h3(style = "color:#475569; margin-top:16px;", "請上傳 C15 資料檔"),
      tags$p(style = "color:#94a3b8; font-size:14px;",
             "使用左側「上傳 CSV」選擇 c15_enriched.csv"),
      tags$p(style = "color:#94a3b8; font-size:12px;",
             "資料僅在本次 session 暫存，關閉瀏覽器後自動清除")
    )
  })

  # ── Data loading ────────────────────────────────────────────────────────

  load_into_rv <- function(path) {
    full     <- load_data(path)
    rv$full  <- full
    rv$data  <- full[, intersect(KEY_COLS, names(full)), drop=FALSE]
    rv$selected <- NULL
  }

  observe({ req(is.null(rv$full)); if (AUTO_LOAD) load_into_rv(DATA_PATH) })

  observeEvent(input$upload, {
    req(input$upload)
    load_into_rv(input$upload$datapath)
    rv$log <- rv$log[0, ]
    showNotification("已載入上傳 CSV", type="message")
  })

  observeEvent(input$reload, {
    load_into_rv(DATA_PATH)
    rv$log <- rv$log[0, ]
    showNotification("已重新載入預設資料", type="message")
  })

  # ── Filter / column controls ────────────────────────────────────────────

  observeEvent(input$filter_col, {
    req(rv$data)
    col <- input$filter_col
    if (col == "(全部)" || !col %in% names(rv$data)) {
      updateSelectInput(session, "filter_val",
                        choices = c("(全部)", "NA / 缺失"))
    } else {
      vals <- sort(unique(as.character(rv$data[[col]])))
      updateSelectInput(session, "filter_val",
                        choices = c("(全部)", "NA / 缺失",
                                    vals[!vals %in% c("NA","")]))
    }
  })

  filtered_rows <- reactive({
    req(rv$data)
    rows <- seq_len(nrow(rv$data))
    col  <- input$filter_col; val <- input$filter_val
    if (!is.null(col) && col != "(全部)" && col %in% names(rv$data)) {
      if (!is.null(val) && val == "NA / 缺失")
        rows <- which(is.na(rv$data[[col]]))
      else if (!is.null(val) && val != "(全部)")
        rows <- which(as.character(rv$data[[col]]) == val)
    }
    rows
  })

  show_cols <- reactive({
    req(input$show_cols)
    intersect(input$show_cols, names(rv$data))
  })

  # ── Missing overview ─────────────────────────────────────────────────────

  output$miss_bar <- renderPlot({
    req(rv$data)
    df   <- rv$data
    pcts <- sapply(names(df), function(c) pct_complete(df[[c]]))
    lmap <- setNames(COL_LABELS, KEY_COLS)
    tbl  <- data.frame(
      label = ifelse(names(pcts) %in% names(lmap), lmap[names(pcts)], names(pcts)),
      pct   = pcts,
      fill  = ifelse(pcts >= 90, "#22c55e", ifelse(pcts >= 70, "#f59e0b", "#ef4444"))
    )
    tbl$label <- factor(tbl$label, levels = tbl$label[order(pcts)])
    ggplot(tbl, aes(x=label, y=pct, fill=fill)) +
      geom_col(width=0.7) +
      geom_text(aes(label=paste0(pct,"%")), hjust=-0.1, size=3.2) +
      geom_hline(yintercept=c(70,90), linetype="dashed",
                 colour=c("#f59e0b","#22c55e"), linewidth=0.5) +
      scale_fill_identity() +
      scale_y_continuous(limits=c(0,112), labels=function(x) paste0(x,"%")) +
      coord_flip() +
      labs(x=NULL, y="完整率 (%)") +
      theme_minimal(base_size=12) +
      theme(panel.grid.minor=element_blank(), panel.grid.major.y=element_blank())
  })

  output$miss_map <- renderPlot({
    req(rv$data)
    df <- rv$data[filtered_rows(), show_cols(), drop=FALSE]
    if (nrow(df) == 0) return(NULL)
    df_c <- as.data.frame(lapply(df, function(x) {
      x <- as.character(x); x[x %in% c("NA","")] <- NA_character_; x
    }), stringsAsFactors=FALSE)
    naniar::vis_miss(df_c, warn_large_data=FALSE) +
      labs(title=paste0("缺失模式 (", nrow(df), " 列)")) +
      theme(axis.text.x=element_text(angle=45, hjust=1, size=9))
  })

  # ── Editor table (read-only; row-click selects) ──────────────────────────

  output$miss_badge <- renderUI({
    req(rv$data)
    df  <- rv$data[, show_cols(), drop=FALSE]
    na  <- sum(sapply(df, function(x) sum(is.na(x))))
    pct <- round(100 * na / (nrow(df) * ncol(df)), 1)
    tags$span(class=if(pct>20)"badge-miss" else "badge-ok",
              paste0("缺失: ", na, " 格 (", pct, "%)"))
  })

  output$row_info <- renderUI({
    req(rv$data)
    sel <- rv$selected
    if (is.null(sel)) {
      tags$span(class="row-info", "← 點擊表格列以選取；再按「編輯選取列」")
    } else {
      pid <- rv$data[["病歷號(2)"]][sel]
      tags$span(class="row-info",
                tags$b(paste0("已選取第 ", sel, " 列")),
                " · 病歷號: ", tags$code(pid))
    }
  })

  output$editor_tbl <- renderDT({
    req(rv$data)
    rows <- filtered_rows()
    cols <- show_cols()
    df   <- rv$data[rows, cols, drop=FALSE]

    # Mark NA cells with italic red placeholder text via JS
    datatable(
      df,
      rownames  = FALSE,
      selection = "single",
      editable  = FALSE,
      filter    = "top",
      extensions = "Scroller",
      options   = list(
        scrollX     = TRUE,
        scrollY     = "460px",
        scroller    = TRUE,
        deferRender = TRUE,
        pageLength  = 50,
        dom         = "lfrtip",
        columnDefs  = list(list(width="90px", targets="_all")),
        rowCallback = JS("
          function(row, data) {
            $('td', row).each(function() {
              var txt = $(this).text().trim();
              if (txt === '' || txt === 'NA') {
                $(this).html('<span style=\"color:#dc2626;font-style:italic;\">NA</span>');
                $(this).css('background-color', '#fff1f2');
              }
            });
          }
        ")
      )
    )
  }, server = TRUE)

  # Track row selection → map back to real row in rv$data
  observeEvent(input$editor_tbl_rows_selected, {
    sel <- input$editor_tbl_rows_selected
    if (length(sel) == 0) { rv$selected <- NULL; return() }
    rv$selected <- filtered_rows()[sel]
  })

  observeEvent(input$clear_sel, {
    rv$selected <- NULL
    proxy <- dataTableProxy("editor_tbl")
    selectRows(proxy, NULL)
  })

  # ── Edit modal ───────────────────────────────────────────────────────────

  observeEvent(input$open_edit, {
    sel <- rv$selected
    if (is.null(sel)) {
      showNotification("請先點選表格中的一列", type="warning"); return()
    }

    row  <- rv$data[sel, , drop=FALSE]
    pid  <- row[["病歷號(2)"]]
    sub  <- row[["subsite"]]; hist <- row[["histology_group"]]

    # Build two-column form layout grouped by topic
    modal_form <- tagList(
      # Read-only context strip
      div(style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;
                 padding:8px 14px;margin-bottom:14px;font-size:12px;",
        tags$b("病歷號: "), tags$code(pid), "  │  ",
        tags$b("Subsite: "), sub, "  │  ",
        tags$b("Histology: "), hist
      ),

      # Group: Demographics
      div(class="modal-section",
        div(class="modal-section-title", "人口學"),
        fluidRow(
          column(4, build_modal_input(EDIT_FIELDS[[1]], row[["sex"]])),
          column(4, build_modal_input(EDIT_FIELDS[[2]], row[["age"]])),
          column(4, build_modal_input(EDIT_FIELDS[[3]], row[["bmi"]]))
        )
      ),

      # Group: Tumour
      div(class="modal-section",
        div(class="modal-section-title", "腫瘤特性"),
        fluidRow(
          column(12, build_modal_input(EDIT_FIELDS[[4]], row[["grade_label"]]))
        )
      ),

      # Group: Staging
      div(class="modal-section",
        div(class="modal-section-title", "分期"),
        fluidRow(
          column(3, build_modal_input(EDIT_FIELDS[[5]], row[["clin_stage"]])),
          column(3, build_modal_input(EDIT_FIELDS[[6]], row[["path_stage"]])),
          column(3, build_modal_input(EDIT_FIELDS[[7]], row[["stage"]])),
          column(3, build_modal_input(EDIT_FIELDS[[8]], row[["stage_group"]]))
        )
      ),

      # Group: Treatment
      div(class="modal-section",
        div(class="modal-section-title", "治療"),
        fluidRow(
          column(4, build_modal_input(EDIT_FIELDS[[9]],  row[["surgery"]])),
          column(4, build_modal_input(EDIT_FIELDS[[10]], row[["radiation"]])),
          column(4, build_modal_input(EDIT_FIELDS[[11]], row[["chemo"]]))
        )
      ),

      # Group: Lifestyle
      div(class="modal-section",
        div(class="modal-section-title", "生活習慣"),
        fluidRow(
          column(4, build_modal_input(EDIT_FIELDS[[12]], row[["alcohol"]])),
          column(4, build_modal_input(EDIT_FIELDS[[13]], row[["smoker"]])),
          column(4, build_modal_input(EDIT_FIELDS[[14]], row[["betel_nut"]]))
        )
      ),

      # Group: Outcome
      div(class="modal-section",
        div(class="modal-section-title", "預後"),
        fluidRow(
          column(6, build_modal_input(EDIT_FIELDS[[15]], row[["vital_status"]])),
          column(6, build_modal_input(EDIT_FIELDS[[16]], row[["os_days"]]))
        )
      )
    )

    showModal(modalDialog(
      title = tags$span(icon("pen"), paste0(" 編輯列 ", sel, "  ·  病歷號: ", pid)),
      size  = "l",
      modal_form,
      footer = tagList(
        actionButton("save_edit", "儲存變更", icon=icon("check"),
                     class="btn-success"),
        modalButton("取消", icon=icon("xmark"))
      )
    ))
  })

  # ── Save modal edits ─────────────────────────────────────────────────────

  observeEvent(input$save_edit, {
    sel <- rv$selected
    req(!is.null(sel))

    pid_col <- "病歷號(2)"
    pid     <- rv$data[[pid_col]][sel]
    n_changed <- 0

    for (fld in EDIT_FIELDS) {
      raw_new <- input[[fld$id]]

      new_val <-
        if (is.null(raw_new) || length(raw_new) == 0)       NA_character_
        else if (raw_new == NA_SENTINEL)                     NA_character_
        else if (fld$type == "numeric" && is.na(suppressWarnings(as.numeric(raw_new))))
                                                             NA_character_
        else as.character(raw_new)

      old_val <- as.character(rv$data[[fld$col]][sel])
      old_val <- if (is.na(rv$data[[fld$col]][sel])) NA_character_ else old_val

      changed <- !identical(
        ifelse(is.na(old_val), NA_character_, old_val),
        ifelse(is.na(new_val), NA_character_, new_val)
      )

      if (changed) {
        rv$data[[fld$col]][sel] <- new_val
        rv$full[[fld$col]][sel] <- new_val
        n_changed <- n_changed + 1

        rv$log <- rbind(rv$log, data.frame(
          Time     = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
          Row      = sel,
          ID       = as.character(pid),
          Column   = fld$col,
          OldValue = ifelse(is.na(old_val), "NA", old_val),
          NewValue = ifelse(is.na(new_val), "NA", new_val),
          stringsAsFactors = FALSE
        ))
      }
    }

    removeModal()

    if (n_changed > 0) {
      # Refresh table
      proxy <- dataTableProxy("editor_tbl")
      replaceData(proxy,
                  rv$data[filtered_rows(), show_cols(), drop=FALSE],
                  resetPaging=FALSE, rownames=FALSE)
      showNotification(paste0("已更新 ", n_changed, " 個欄位（列 ", sel, "）"),
                       type="message", duration=4)
    } else {
      showNotification("無變更", type="warning", duration=2)
    }
  })

  output$edit_count_msg <- renderUI({
    n <- nrow(rv$log)
    if (n == 0) return(NULL)
    tags$div(style="margin-top:6px;color:#6b7280;font-size:12px;",
             paste0("本次共 ", n, " 筆變更"))
  })

  # ── Changelog ────────────────────────────────────────────────────────────

  output$changelog_tbl <- renderDT({
    datatable(rv$log, rownames=FALSE, class="changelog-tbl",
              options=list(pageLength=20, dom="tp",
                           order=list(list(0,"desc"))))
  })

  output$edit_stats <- renderText({
    req(rv$data); df <- rv$data
    paste0(
      "已編輯 ", nrow(rv$log), " 筆\n",
      "stage_group: ", pct_complete(df$stage_group), "%\n",
      "clin_stage:  ", pct_complete(df$clin_stage),  "%\n",
      "grade_label: ", pct_complete(df$grade_label), "%\n",
      "alcohol:     ", pct_complete(df$alcohol),      "%\n",
      "smoker:      ", pct_complete(df$smoker),       "%\n",
      "betel_nut:   ", pct_complete(df$betel_nut),    "%\n",
      "bmi:         ", pct_complete(df$bmi),          "%"
    )
  })

  # ── Downloads ────────────────────────────────────────────────────────────

  output$dl_edited <- downloadHandler(
    filename = function() paste0("c15_edited_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content  = function(f) write_csv(rv$full, f)
  )
  output$dl_changelog <- downloadHandler(
    filename = function() paste0("c15_changelog_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"),
    content  = function(f) write_csv(rv$log, f)
  )
}

shinyApp(ui, server)
