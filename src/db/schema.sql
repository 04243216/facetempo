-- FaceTempo 数据库结构（MySQL 8.x）
-- 三张表：视频/样本、逐帧特征、诊断结果。
-- 用法：mysql -u root -p < src/db/schema.sql
--
-- 说明：此处为骨架设计，W4 起按实际指标字段微调。

CREATE DATABASE IF NOT EXISTS facetempo
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE facetempo;

-- 1) 样本：一条视频 = 一个样本
CREATE TABLE IF NOT EXISTS videos (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name          VARCHAR(255)    NOT NULL COMMENT '文件名，不含路径',
    source        ENUM('real', 'ai', 'test') NOT NULL,
    tool          VARCHAR(64)     NULL COMMENT 'AI 片段使用的工具，real/test 为 NULL',
    fps           DECIMAL(6, 3)   NULL,
    frame_count   INT UNSIGNED    NULL,
    duration_sec  DECIMAL(8, 3)   NULL,
    sha256        CHAR(64)        NULL COMMENT '文件哈希，用于去重',
    captured_at   DATETIME        NULL,
    notes         TEXT            NULL,
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_videos_sha256 (sha256),
    KEY idx_videos_source_tool (source, tool)
) ENGINE = InnoDB COMMENT = '视频样本元信息';

-- 2) 逐帧特征：对齐后的几何 + AU 特征
CREATE TABLE IF NOT EXISTS frame_features (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    video_id      BIGINT UNSIGNED NOT NULL,
    frame_index   INT UNSIGNED    NOT NULL,
    timestamp_sec DECIMAL(10, 4)  NOT NULL,
    features      JSON            NOT NULL COMMENT '该帧全部特征，键名与 docs/metrics_spec.md 一致',
    face_found    TINYINT(1)      NOT NULL DEFAULT 1,
    PRIMARY KEY (id),
    UNIQUE KEY uk_frame_video_frame (video_id, frame_index),
    CONSTRAINT fk_frame_video FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
) ENGINE = InnoDB COMMENT = '逐帧特征（宽表放不下时用 JSON 兜底）';

-- 3) 诊断结果：一个样本一次诊断 = 一行
CREATE TABLE IF NOT EXISTS diagnoses (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    video_id      BIGINT UNSIGNED NOT NULL,
    metrics       JSON            NOT NULL COMMENT '六项指标值',
    model_version VARCHAR(64)     NOT NULL COMMENT '计算脚本/参数版本，便于复现',
    report_path   VARCHAR(512)    NULL COMMENT 'outputs/reports/ 下相对路径',
    created_at    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_diag_video (video_id),
    CONSTRAINT fk_diag_video FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
) ENGINE = InnoDB COMMENT = '诊断结果';
