-- 1. Trainsets ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trainset (
   id            SERIAL PRIMARY KEY,
   version       VARCHAR(20)    NOT NULL,
   description   VARCHAR(1000),
   base_model_inst_id INTEGER,
   new_model_inst_id  INTEGER,
   created_by    VARCHAR(100),
   is_active     BOOLEAN        DEFAULT TRUE,
   create_dt     TIMESTAMPTZ    DEFAULT NOW(),
   last_run      TIMESTAMPTZ
);

-- 2. Acronyms -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS acronyms (
   id           SERIAL PRIMARY KEY,
   trainset_id  INTEGER        NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   acronym_en   VARCHAR(20)    NOT NULL,
   acronym_fr   VARCHAR(20),
   text_en      VARCHAR(500)   NOT NULL,
   text_fr      VARCHAR(500),
   example_en   TEXT,
   example_fr   TEXT,
   data_status  INTEGER        DEFAULT 1,
   create_dt    TIMESTAMPTZ    DEFAULT NOW(),
   update_dt    TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS acronyms_unique_trainset_idx
  ON acronyms (trainset_id, acronym_en);

-- 3. File Storage -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trainset_files (
   id          SERIAL PRIMARY KEY,
   trainset_id INTEGER       NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   path_en     VARCHAR(1000) NOT NULL,
   path_fr     VARCHAR(1000),
   content_en  TEXT,
   content_fr  TEXT,
   create_dt   TIMESTAMPTZ   DEFAULT NOW()
);

-- 4. Sentence Extraction ------------------------------------------------------
CREATE TABLE IF NOT EXISTS trainset_sentences (
   id              SERIAL PRIMARY KEY,
   trainset_id     INTEGER       NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   file_id         INTEGER       NOT NULL REFERENCES trainset_files(id) ON DELETE CASCADE,
   sentence_en     VARCHAR(1000) NOT NULL,
   sentence_fr     VARCHAR(1000),
   sentence_laser_score FLOAT
);
CREATE INDEX IF NOT EXISTS trainset_sentences_laser_idx
  ON trainset_sentences (sentence_laser_score);

-- 5. Acronym ↔ Sentence Mapping ----------------------------------------------
-- The explicit trainset_id + CHECK constraint removed (can't use sub‑queries in
-- CHECK). Integrity is naturally preserved because:
--   * acronym_id → acronyms (FK) → trainset_id
--   * sentence_id → trainset_sentences (FK) → trainset_id
-- A mismatched pair would violate at the application layer but not DB layer.
CREATE TABLE IF NOT EXISTS acronym_sentence_matches (
   id          SERIAL PRIMARY KEY,
   acronym_id  INTEGER    NOT NULL REFERENCES acronyms(id) ON DELETE CASCADE,
   sentence_id INTEGER    NOT NULL REFERENCES trainset_sentences(id) ON DELETE CASCADE,
   match_score FLOAT,
   created_dt  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS acronym_sentence_match_idx
  ON acronym_sentence_matches (acronym_id, sentence_id);

-- 6. Training Data (Human / Generated) ---------------------------------------
CREATE TABLE IF NOT EXISTS acronym_traindata (
   id           SERIAL PRIMARY KEY,
   trainset_id  INTEGER     NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   acronym_id   INTEGER     NOT NULL REFERENCES acronyms(id) ON DELETE CASCADE,
   provided_by  VARCHAR(20),
   generated_bytrainset_id INTEGER,
   text_en      TEXT,
   text_fr      TEXT,
   reason       VARCHAR(500),
   create_dt    TIMESTAMPTZ DEFAULT NOW()
);

-- 7. Models -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
   id         SERIAL PRIMARY KEY,
   name       VARCHAR(100) NOT NULL,
   version    VARCHAR(20)  NOT NULL,
   department VARCHAR(100),
   score      FLOAT,
   checkpoint INTEGER,
   status     VARCHAR(50)  DEFAULT 'created',
   create_dt  TIMESTAMPTZ  DEFAULT NOW(),
   UNIQUE(name, version)
);

-- 8. Test Sets ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS testset (
   id          SERIAL PRIMARY KEY,
   trainset_id INTEGER       NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   version     VARCHAR(20)   NOT NULL,
   description VARCHAR(1000),
   created_by  VARCHAR(100),
   is_active   BOOLEAN       DEFAULT TRUE,
   create_dt   TIMESTAMPTZ   DEFAULT NOW(),
   last_run    TIMESTAMPTZ,
   base_model_inst_id INTEGER,
   new_model_inst_id  INTEGER
);

CREATE TABLE IF NOT EXISTS testset_sentences (
   id              SERIAL PRIMARY KEY,
   testset_id      INTEGER       NOT NULL REFERENCES testset(id) ON DELETE CASCADE,
   sentence_en     VARCHAR(1000) NOT NULL,
   sentence_fr     VARCHAR(1000),
   sentence_laser_score FLOAT
);

CREATE TABLE IF NOT EXISTS testset_contents (
   id           SERIAL PRIMARY KEY,
   testset_id   INTEGER NOT NULL REFERENCES testset(id) ON DELETE CASCADE,
   sentence_id  INTEGER NOT NULL REFERENCES testset_sentences(id) ON DELETE CASCADE
);

-- 9. Schedules & Training Runs ----------------------------------------------
CREATE TABLE IF NOT EXISTS trainset_schedule (
   id                SERIAL PRIMARY KEY,
   trainset_id       INTEGER NOT NULL REFERENCES trainset(id) ON DELETE CASCADE,
   testset_id        INTEGER REFERENCES testset(id) ON DELETE SET NULL,
   model_id          INTEGER REFERENCES models(id) ON DELETE SET NULL,
   days_between_runs INTEGER,
   schedule_hour     INTEGER,
   schedule_minute   INTEGER,
   is_active         BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS trainset_training_runs (
   id                SERIAL PRIMARY KEY,
   schedule_id       INTEGER NOT NULL REFERENCES trainset_schedule(id) ON DELETE CASCADE,
   scheduled_dt      TIMESTAMPTZ,
   trainset_version  VARCHAR(20),
   testset_version   VARCHAR(20),
   model_version     VARCHAR(100),
   testset_id        INTEGER REFERENCES testset(id) ON DELETE SET NULL,
   score             FLOAT,
   duration          INTEGER,
   status            VARCHAR(20) DEFAULT 'running',
   train_loss        FLOAT,
   val_loss          FLOAT,
   val_accuracy      FLOAT,
   cumulative_accuracy FLOAT,
   total_correct     INTEGER,
   total_samples     INTEGER,
   model_path        VARCHAR(200),
   is_model_loaded   BOOLEAN,
   create_dt         TIMESTAMPTZ DEFAULT NOW(),
   is_active         BOOLEAN     DEFAULT TRUE
);

-- 10. Trainer Schema Version --------------------------------------------------
CREATE TABLE IF NOT EXISTS trainer_schema_version (
    id SERIAL PRIMARY KEY,
    major_version_number INTEGER NOT NULL,
    minor_version_number INTEGER NOT NULL,
    db_script_type_id    INTEGER NOT NULL,
    script_number        INTEGER NOT NULL,
    comment_msg          VARCHAR(4000) NOT NULL,
    UNIQUE(major_version_number, minor_version_number, db_script_type_id, script_number)
);

-- 11. Misc Tables -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS settings (
   id SERIAL PRIMARY KEY,
   settings_str TEXT
);

CREATE TABLE IF NOT EXISTS background_tasks (
   id SERIAL PRIMARY KEY,
   task_key      VARCHAR(50),
   pid           INTEGER,
   process_type  VARCHAR(50),
   model_id      INTEGER,
   status        VARCHAR(50),
   start_time    TIMESTAMPTZ,
   end_time      TIMESTAMPTZ,
   error_message VARCHAR(500)
);


/*
 * Expert GPT Models
 */

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_status') THEN
        CREATE TYPE model_status AS ENUM ('created', 'training', 'deployed', 'failed');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'schedule_status') THEN
        CREATE TYPE schedule_status AS ENUM ('pending', 'running', 'completed', 'failed', 'paused');
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'raw_data_status') THEN
        CREATE TYPE raw_data_status AS ENUM ('uploaded', 'processing', 'processed', 'failed');
    END IF;
END
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS expert_gpt_models(
 id VARCHAR(100) PRIMARY KEY,
 model_path TEXT NOT NULL,
 model_name VARCHAR(100) NOT NULL,
 status model_status DEFAULT 'created' NOT NULL,
 details TEXT,
 is_every_document_in_knowledge SMALLINT NOT NULL CHECK(is_every_document_in_knowledge IN (0,1)),
 scheduled_dt TIMESTAMPTZ,
 schedule_status schedule_status,
 schedule_details TEXT,
 created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
 updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- recreate trigger if it already exists
DROP TRIGGER IF EXISTS trg_set_updated_at ON expert_gpt_models;
CREATE TRIGGER trg_set_updated_at
BEFORE UPDATE ON expert_gpt_models
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS expert_gpt_schedules(
 id SERIAL PRIMARY KEY,
 model_id VARCHAR(100) REFERENCES expert_gpt_models(id) ON DELETE CASCADE,
 scheduled_dt TIMESTAMPTZ NOT NULL,
 is_active SMALLINT NOT NULL CHECK(is_active IN (0,1)),
 created_dt TIMESTAMPTZ DEFAULT NOW() NOT NULL,
 next_run_dt TIMESTAMPTZ
);
--DROP TABLE IF EXISTS raw_train_data CASCADE;
CREATE TABLE IF NOT EXISTS raw_train_data(
 id SERIAL PRIMARY KEY,
 path TEXT NOT NULL,
 question TEXT,
 answer TEXT,
 params JSONB,
 task TEXT,
 created_dt TIMESTAMPTZ DEFAULT NOW() NOT NULL,
 expert_gpt_id VARCHAR(100) REFERENCES expert_gpt_models(id) ON DELETE CASCADE,
 status raw_data_status DEFAULT 'uploaded' NOT NULL,
 details TEXT
);

CREATE OR REPLACE VIEW expert_gpt_model_list AS SELECT * FROM expert_gpt_models;
CREATE OR REPLACE VIEW raw_train_data_list AS SELECT * FROM raw_train_data;

CREATE TABLE IF NOT EXISTS jobs (
    job_id     VARCHAR(120) PRIMARY KEY,
    payload    JSONB         NOT NULL,
    status     VARCHAR(32)   NOT NULL,
    submitted  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    progress   JSONB         NOT NULL DEFAULT '{}'::jsonb,
    last_msg   TEXT,
    updated    TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- keep updated column fresh on every change
CREATE OR REPLACE FUNCTION trg_jobs_set_updated() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_updated ON jobs;
CREATE TRIGGER trg_jobs_updated
BEFORE UPDATE ON jobs
FOR EACH ROW EXECUTE FUNCTION trg_jobs_set_updated();
/*
End of Expert GPT Models
*/



