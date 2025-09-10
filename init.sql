-- Create UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables matching the Go application schema
CREATE TABLE IF NOT EXISTS menu (
    id UUID PRIMARY KEY,
    original_image_url TEXT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','PROCESSING','COMPLETE','FAILED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    failure_reason TEXT NULL
);

CREATE TABLE IF NOT EXISTS dish (
    id UUID PRIMARY KEY,
    menu_id UUID NOT NULL REFERENCES menu(id) ON DELETE CASCADE,
    section TEXT NULL,
    name TEXT NOT NULL,
    description TEXT NULL,
    price_raw TEXT NULL,
    price_value NUMERIC NULL,
    tags TEXT[] NULL,
    sort_order INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dish_image (
    dish_id UUID PRIMARY KEY REFERENCES dish(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    width INT NULL,
    height INT NULL,
    format TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_menu_status ON menu(status);
CREATE INDEX IF NOT EXISTS idx_dish_menu_id_sort ON dish(menu_id, sort_order);

-- Insert sample data for testing
INSERT INTO menu (id, status, created_at, updated_at) 
VALUES (uuid_generate_v4(), 'COMPLETE', now(), now());

-- Get the menu ID for sample dishes
WITH sample_menu AS (
    SELECT id FROM menu LIMIT 1
)
INSERT INTO dish (id, menu_id, section, name, description, price_raw, tags, sort_order)
SELECT 
    uuid_generate_v4(),
    sample_menu.id,
    'Appetizers',
    'Caesar Salad',
    'Fresh romaine lettuce with parmesan cheese and croutons',
    '$12.99',
    ARRAY['salad', 'vegetarian'],
    1
FROM sample_menu;
