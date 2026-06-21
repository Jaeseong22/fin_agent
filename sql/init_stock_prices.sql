CREATE TABLE IF NOT EXISTS stock_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,

    name VARCHAR(100) NOT NULL,
    official_name VARCHAR(100) NOT NULL,
    trade_date DATE NOT NULL,

    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT UNSIGNED,

    market_cap BIGINT,

    UNIQUE KEY uq_stock_date (name, trade_date),
    INDEX idx_trade_date (trade_date),
    INDEX idx_name (name),
    INDEX idx_official_name (official_name),
    INDEX idx_market_cap (market_cap)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS market_cap_stage (
    name VARCHAR(100) NOT NULL,
    trade_date DATE NOT NULL,
    market_cap BIGINT,
    UNIQUE KEY uq_stage (name, trade_date),
    INDEX idx_stage_name_date (name, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;