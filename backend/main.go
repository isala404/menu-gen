package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/lib/pq"
	"github.com/rs/cors"
)

// Configuration
type Config struct {
	Port                string
	DatabaseURL         string
	OpenAIAPIKey        string
	OpenAIServiceURL    string
	ReplicateAPIKey     string
	ReplicateServiceURL string
	MaxConcurrentImages int
	MaxImageSize        int64
	ProcessingTimeout   time.Duration
}

// Domain Models
type MenuStatus string

const (
	StatusPending    MenuStatus = "PENDING"
	StatusProcessing MenuStatus = "PROCESSING"
	StatusComplete   MenuStatus = "COMPLETE"
	StatusFailed     MenuStatus = "FAILED"
)

type Menu struct {
	ID               string     `json:"menuId" db:"id"`
	OriginalImageURL *string    `json:"-" db:"original_image_url"`
	Status           MenuStatus `json:"status" db:"status"`
	CreatedAt        time.Time  `json:"-" db:"created_at"`
	UpdatedAt        time.Time  `json:"-" db:"updated_at"`
	FailureReason    *string    `json:"error,omitempty" db:"failure_reason"`
	Sections         []Section  `json:"sections,omitempty"`
	GeneratedAt      *time.Time `json:"generatedAt,omitempty"`
}

type Section struct {
	Name   string `json:"name"`
	Dishes []Dish `json:"dishes"`
}

type Dish struct {
	ID          string   `json:"id" db:"id"`
	MenuID      string   `json:"-" db:"menu_id"`
	Section     *string  `json:"-" db:"section"`
	Name        string   `json:"name" db:"name"`
	Description *string  `json:"description,omitempty" db:"description"`
	Price       *string  `json:"price,omitempty" db:"price_raw"`
	Tags        []string `json:"tags,omitempty" db:"tags"`
	SortOrder   int      `json:"-" db:"sort_order"`
	Image       *Image   `json:"image,omitempty"`
}

type Image struct {
	URL string `json:"url"`
}

type DishImage struct {
	DishID    string    `db:"dish_id"`
	ImageURL  string    `db:"image_url"`
	Width     *int      `db:"width"`
	Height    *int      `db:"height"`
	Format    *string   `db:"format"`
	CreatedAt time.Time `db:"created_at"`
}

// OpenAI Types
type OpenAIRequest struct {
	Model          string    `json:"model"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name   string      `json:"name"`
			Schema interface{} `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
	MaxTokens int `json:"max_tokens"`
}

type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type OpenAIResponse struct {
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

// LLM Response Schema
type LLMMenuResponse struct {
	Sections   []LLMSection `json:"sections"`
	Properties *struct {
		Sections []LLMSection `json:"sections"`
	} `json:"properties,omitempty"`
}

type LLMSection struct {
	Name   string    `json:"name"`
	Dishes []LLMDish `json:"dishes"`
}

type LLMDish struct {
	Name        string   `json:"name"`
	Description *string  `json:"description"`
	Price       *string  `json:"price"`
	Tags        []string `json:"tags"`
}

// Replicate Types
type ReplicateRequest struct {
	Input ReplicateInput `json:"input"`
}

type ReplicateInput struct {
	Prompt               string  `json:"prompt"`
	AspectRatio          string  `json:"aspect_ratio"`
	NumOutputs           int     `json:"num_outputs"`
	NumInferenceSteps    int     `json:"num_inference_steps"`
	Guidance             float64 `json:"guidance"`
	OutputFormat         string  `json:"output_format"`
	OutputQuality        int     `json:"output_quality"`
	GoFast               bool    `json:"go_fast"`
	DisableSafetyChecker bool    `json:"disable_safety_checker"`
}

type ReplicateResponse struct {
	ID     string        `json:"id"`
	Status string        `json:"status"`
	Output []string      `json:"output"`
	URLs   ReplicateURLs `json:"urls"`
	Error  *string       `json:"error"`
}

type ReplicateURLs struct {
	Get string `json:"get"`
}

// Database Repository
type Repository struct {
	db *sql.DB
}

func NewRepository(db *sql.DB) *Repository {
	return &Repository{db: db}
}

func (r *Repository) CreateMenu(ctx context.Context, menu *Menu) error {
	query := `
		INSERT INTO menu (id, original_image_url, status, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5)
	`
	now := time.Now()
	_, err := r.db.ExecContext(ctx, query, menu.ID, menu.OriginalImageURL, menu.Status, now, now)
	if err != nil {
		return fmt.Errorf("failed to create menu: %w", err)
	}
	menu.CreatedAt = now
	menu.UpdatedAt = now
	return nil
}

func (r *Repository) UpdateMenuStatus(ctx context.Context, menuID string, status MenuStatus, failureReason *string) error {
	query := `
		UPDATE menu 
		SET status = $1, failure_reason = $2, updated_at = $3 
		WHERE id = $4
	`
	_, err := r.db.ExecContext(ctx, query, status, failureReason, time.Now(), menuID)
	if err != nil {
		return fmt.Errorf("failed to update menu status: %w", err)
	}
	return nil
}

func (r *Repository) GetMenu(ctx context.Context, menuID string) (*Menu, error) {
	menu := &Menu{}
	query := `
		SELECT id, original_image_url, status, created_at, updated_at, failure_reason
		FROM menu WHERE id = $1
	`
	err := r.db.QueryRowContext(ctx, query, menuID).Scan(
		&menu.ID, &menu.OriginalImageURL, &menu.Status,
		&menu.CreatedAt, &menu.UpdatedAt, &menu.FailureReason,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("menu not found")
		}
		return nil, fmt.Errorf("failed to get menu: %w", err)
	}

	if menu.Status == StatusComplete {
		dishes, err := r.GetDishesWithImages(ctx, menuID)
		if err != nil {
			return nil, fmt.Errorf("failed to get dishes: %w", err)
		}
		menu.Sections = r.groupDishesBySection(dishes)
		menu.GeneratedAt = &menu.UpdatedAt
	}

	return menu, nil
}

func (r *Repository) CreateDishes(ctx context.Context, dishes []Dish) error {
	if len(dishes) == 0 {
		return nil
	}

	query := `
		INSERT INTO dish (id, menu_id, section, name, description, price_raw, tags, sort_order, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`

	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	for _, dish := range dishes {
		now := time.Now()
		_, err := tx.ExecContext(ctx, query,
			dish.ID, dish.MenuID, dish.Section, dish.Name,
			dish.Description, dish.Price, pq.Array(dish.Tags),
			dish.SortOrder, now,
		)
		if err != nil {
			return fmt.Errorf("failed to create dish: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit dishes: %w", err)
	}

	return nil
}

func (r *Repository) CreateDishImage(ctx context.Context, dishImage *DishImage) error {
	query := `
		INSERT INTO dish_image (dish_id, image_url, width, height, format, created_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	now := time.Now()
	_, err := r.db.ExecContext(ctx, query,
		dishImage.DishID, dishImage.ImageURL, dishImage.Width,
		dishImage.Height, dishImage.Format, now,
	)
	if err != nil {
		return fmt.Errorf("failed to create dish image: %w", err)
	}
	return nil
}

func (r *Repository) GetDishesWithImages(ctx context.Context, menuID string) ([]Dish, error) {
	query := `
		SELECT 
			d.id, d.menu_id, d.section, d.name, d.description, d.price_raw, d.tags, d.sort_order,
			di.image_url
		FROM dish d
		LEFT JOIN dish_image di ON d.id = di.dish_id
		WHERE d.menu_id = $1
		ORDER BY d.sort_order, d.created_at
	`

	rows, err := r.db.QueryContext(ctx, query, menuID)
	if err != nil {
		return nil, fmt.Errorf("failed to query dishes: %w", err)
	}
	defer rows.Close()

	var dishes []Dish
	for rows.Next() {
		var dish Dish
		var imageURL *string
		var tags pq.StringArray

		err := rows.Scan(
			&dish.ID, &dish.MenuID, &dish.Section, &dish.Name,
			&dish.Description, &dish.Price, &tags, &dish.SortOrder,
			&imageURL,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan dish: %w", err)
		}

		dish.Tags = []string(tags)
		if imageURL != nil {
			dish.Image = &Image{URL: *imageURL}
		}

		dishes = append(dishes, dish)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating dishes: %w", err)
	}

	return dishes, nil
}

func (r *Repository) groupDishesBySection(dishes []Dish) []Section {
	sectionMap := make(map[string][]Dish)
	var sectionOrder []string

	for _, dish := range dishes {
		section := "Main"
		if dish.Section != nil && *dish.Section != "" {
			section = *dish.Section
		}

		if _, exists := sectionMap[section]; !exists {
			sectionOrder = append(sectionOrder, section)
		}
		sectionMap[section] = append(sectionMap[section], dish)
	}

	var sections []Section
	for _, sectionName := range sectionOrder {
		sections = append(sections, Section{
			Name:   sectionName,
			Dishes: sectionMap[sectionName],
		})
	}

	return sections
}

// Services
type MenuService struct {
	repo            *Repository
	openAIClient    *OpenAIClient
	replicateClient *ReplicateClient
	config          *Config
}

func NewMenuService(repo *Repository, openAIClient *OpenAIClient, replicateClient *ReplicateClient, config *Config) *MenuService {
	return &MenuService{
		repo:            repo,
		openAIClient:    openAIClient,
		replicateClient: replicateClient,
		config:          config,
	}
}

func (s *MenuService) CreateMenu(ctx context.Context, imageData []byte) (*Menu, error) {
	menu := &Menu{
		ID:     uuid.New().String(),
		Status: StatusPending,
	}

	if err := s.repo.CreateMenu(ctx, menu); err != nil {
		return nil, fmt.Errorf("failed to create menu: %w", err)
	}

	// Start async processing
	go s.processMenu(context.Background(), menu.ID, imageData)

	return menu, nil
}

func (s *MenuService) GetMenu(ctx context.Context, menuID string) (*Menu, error) {
	return s.repo.GetMenu(ctx, menuID)
}

func (s *MenuService) processMenu(ctx context.Context, menuID string, imageData []byte) {
	log.Printf("Starting menu processing for ID: %s", menuID)

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, s.config.ProcessingTimeout)
	defer cancel()

	// Update status to processing
	if err := s.repo.UpdateMenuStatus(ctx, menuID, StatusProcessing, nil); err != nil {
		log.Printf("Failed to update menu status to processing: %v", err)
		return
	}

	// Step 1: Extract menu structure using OpenAI
	log.Printf("Extracting menu structure for ID: %s", menuID)
	llmResponse, err := s.extractMenuStructure(ctx, imageData)
	if err != nil {
		log.Printf("Failed to extract menu structure: %v", err)
		failureReason := "Failed to extract menu structure from image"
		s.repo.UpdateMenuStatus(ctx, menuID, StatusFailed, &failureReason)
		return
	}

	// Step 2: Create dishes in database
	log.Printf("Creating %d dishes for menu ID: %s", s.countDishes(llmResponse), menuID)
	dishes := s.convertLLMResponseToDishes(llmResponse, menuID)
	if err := s.repo.CreateDishes(ctx, dishes); err != nil {
		log.Printf("Failed to create dishes: %v", err)
		failureReason := "Failed to save menu data"
		s.repo.UpdateMenuStatus(ctx, menuID, StatusFailed, &failureReason)
		return
	}

	// Step 3: Generate images for dishes (with bounded concurrency)
	log.Printf("Generating images for dishes for menu ID: %s", menuID)
	if err := s.generateDishImages(ctx, dishes); err != nil {
		log.Printf("Failed to generate some dish images: %v", err)
		// Continue anyway - partial failure in image generation shouldn't fail the whole menu
	}

	// Step 4: Mark as complete
	if err := s.repo.UpdateMenuStatus(ctx, menuID, StatusComplete, nil); err != nil {
		log.Printf("Failed to update menu status to complete: %v", err)
		return
	}

	log.Printf("Successfully completed menu processing for ID: %s", menuID)
}

func (s *MenuService) extractMenuStructure(ctx context.Context, imageData []byte) (*LLMMenuResponse, error) {
	log.Printf("Image data size: %d bytes", len(imageData))

	// Check if we're in test mode (no valid API key)
	if s.openAIClient.apiKey == "sk-your-openai-key-here" || s.openAIClient.apiKey == "" {
		log.Printf("Using test mode - generating sample menu data")
		return s.generateSampleMenuData(), nil
	}

	base64Image := base64.StdEncoding.EncodeToString(imageData)
	imageURL := fmt.Sprintf("data:image/jpeg;base64,%s", base64Image)

	log.Printf("Base64 image URL length: %d", len(imageURL))

	request := OpenAIRequest{
		Model: "gpt-4o",
		Messages: []Message{
			{
				Role: "user",
				Content: []Content{
					{
						Type: "text",
						Text: func() *string {
							text := `Extract menu sections and dishes from this restaurant menu image. 
							For each dish provide: name (required), description (optional), price (optional, preserve currency symbol), and tags (vegan, vegetarian, gluten-free, spicy, etc.).
							Return JSON strictly matching the provided schema. Group dishes by sections (Appetizers, Main Course, etc.).`
							return &text
						}(),
					},
					{
						Type:     "image_url",
						ImageURL: &ImageURL{URL: imageURL},
					},
				},
			},
		},
		ResponseFormat: struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string      `json:"name"`
				Schema interface{} `json:"schema"`
			} `json:"json_schema"`
		}{
			Type: "json_schema",
			JSONSchema: struct {
				Name   string      `json:"name"`
				Schema interface{} `json:"schema"`
			}{
				Name: "menu_extraction",
				Schema: map[string]interface{}{
					"type":     "object",
					"required": []string{"sections"},
					"properties": map[string]interface{}{
						"sections": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{
								"type":     "object",
								"required": []string{"name", "dishes"},
								"properties": map[string]interface{}{
									"name": map[string]interface{}{"type": "string"},
									"dishes": map[string]interface{}{
										"type": "array",
										"items": map[string]interface{}{
											"type":     "object",
											"required": []string{"name"},
											"properties": map[string]interface{}{
												"name":        map[string]interface{}{"type": "string"},
												"description": map[string]interface{}{"type": "string", "nullable": true},
												"price":       map[string]interface{}{"type": "string", "nullable": true},
												"tags": map[string]interface{}{
													"type":  "array",
													"items": map[string]interface{}{"type": "string"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		MaxTokens: 2000,
	}

	response, err := s.openAIClient.CreateChatCompletion(ctx, request)
	if err != nil {
		log.Printf("OpenAI API call failed: %v", err)
		return nil, fmt.Errorf("OpenAI API call failed: %w", err)
	}

	log.Printf("OpenAI response received with %d choices", len(response.Choices))
	if len(response.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenAI response")
	}

	log.Printf("OpenAI response content length: %d", len(response.Choices[0].Message.Content))
	log.Printf("OpenAI response content preview: %.200s", response.Choices[0].Message.Content)

	var llmResponse LLMMenuResponse
	if err := json.Unmarshal([]byte(response.Choices[0].Message.Content), &llmResponse); err != nil {
		log.Printf("Failed to parse LLM response: %v", err)
		log.Printf("Raw response content: %s", response.Choices[0].Message.Content)
		return nil, fmt.Errorf("failed to parse LLM response: %w", err)
	}

	// Handle both direct format {"sections": [...]} and nested format {"properties": {"sections": [...]}}
	sections := llmResponse.Sections
	if len(sections) == 0 && llmResponse.Properties != nil {
		sections = llmResponse.Properties.Sections
	}

	// Create the final response with the correct sections
	finalResponse := &LLMMenuResponse{Sections: sections}

	log.Printf("Parsed LLM response with %d sections", len(finalResponse.Sections))
	for i, section := range finalResponse.Sections {
		log.Printf("Section %d: %s with %d dishes", i, section.Name, len(section.Dishes))
	}

	return finalResponse, nil
}

func (s *MenuService) generateSampleMenuData() *LLMMenuResponse {
	return &LLMMenuResponse{
		Sections: []LLMSection{
			{
				Name: "Appetizers",
				Dishes: []LLMDish{
					{
						Name: "Caesar Salad",
						Description: func() *string {
							s := "Fresh romaine lettuce, parmesan cheese, croutons, and caesar dressing"
							return &s
						}(),
						Price: func() *string { s := "$12.99"; return &s }(),
						Tags:  []string{"vegetarian", "salad"},
					},
					{
						Name:        "Chicken Wings",
						Description: func() *string { s := "Crispy chicken wings with buffalo sauce and blue cheese dip"; return &s }(),
						Price:       func() *string { s := "$15.99"; return &s }(),
						Tags:        []string{"spicy", "chicken"},
					},
				},
			},
			{
				Name: "Main Course",
				Dishes: []LLMDish{
					{
						Name: "Grilled Salmon",
						Description: func() *string {
							s := "Fresh Atlantic salmon with lemon herb seasoning, served with rice and vegetables"
							return &s
						}(),
						Price: func() *string { s := "$24.99"; return &s }(),
						Tags:  []string{"seafood", "healthy"},
					},
					{
						Name:        "Beef Burger",
						Description: func() *string { s := "100% Angus beef patty with lettuce, tomato, onion, and fries"; return &s }(),
						Price:       func() *string { s := "$18.99"; return &s }(),
						Tags:        []string{"beef", "burger"},
					},
					{
						Name:        "Vegetable Pasta",
						Description: func() *string { s := "Penne pasta with seasonal vegetables in marinara sauce"; return &s }(),
						Price:       func() *string { s := "$16.99"; return &s }(),
						Tags:        []string{"vegetarian", "pasta"},
					},
				},
			},
			{
				Name: "Desserts",
				Dishes: []LLMDish{
					{
						Name:        "Chocolate Cake",
						Description: func() *string { s := "Rich chocolate layer cake with chocolate frosting"; return &s }(),
						Price:       func() *string { s := "$8.99"; return &s }(),
						Tags:        []string{"dessert", "chocolate"},
					},
					{
						Name:        "Vanilla Ice Cream",
						Description: func() *string { s := "Premium vanilla ice cream with fresh berries"; return &s }(),
						Price:       func() *string { s := "$6.99"; return &s }(),
						Tags:        []string{"dessert", "cold"},
					},
				},
			},
		},
	}
}

func (s *MenuService) convertLLMResponseToDishes(llmResponse *LLMMenuResponse, menuID string) []Dish {
	var dishes []Dish
	sortOrder := 0

	for _, section := range llmResponse.Sections {
		for _, llmDish := range section.Dishes {
			dish := Dish{
				ID:          uuid.New().String(),
				MenuID:      menuID,
				Section:     &section.Name,
				Name:        llmDish.Name,
				Description: llmDish.Description,
				Price:       llmDish.Price,
				Tags:        llmDish.Tags,
				SortOrder:   sortOrder,
			}
			if dish.Tags == nil {
				dish.Tags = []string{}
			}
			dishes = append(dishes, dish)
			sortOrder++
		}
	}

	return dishes
}

func (s *MenuService) countDishes(llmResponse *LLMMenuResponse) int {
	count := 0
	for _, section := range llmResponse.Sections {
		count += len(section.Dishes)
	}
	return count
}

func (s *MenuService) generateDishImages(ctx context.Context, dishes []Dish) error {
	// Limit concurrent image generation
	semaphore := make(chan struct{}, s.config.MaxConcurrentImages)
	var wg sync.WaitGroup
	var errors []error
	var errorsMutex sync.Mutex

	for _, dish := range dishes {
		wg.Add(1)
		go func(d Dish) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			if err := s.generateSingleDishImage(ctx, d); err != nil {
				errorsMutex.Lock()
				errors = append(errors, fmt.Errorf("failed to generate image for dish %s: %w", d.Name, err))
				errorsMutex.Unlock()
				log.Printf("Failed to generate image for dish %s: %v", d.Name, err)
			} else {
				log.Printf("Successfully generated image for dish: %s", d.Name)
			}
		}(dish)
	}

	wg.Wait()

	if len(errors) > 0 {
		return fmt.Errorf("some image generation failed: %d errors", len(errors))
	}

	return nil
}

func (s *MenuService) generateSingleDishImage(ctx context.Context, dish Dish) error {
	// Retry logic
	maxRetries := 2
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			delay := time.Duration(attempt*500) * time.Millisecond
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		// Generate prompt
		prompt := s.createImagePrompt(dish)

		// Call Replicate API
		imageURL, err := s.replicateClient.GenerateImage(ctx, prompt)
		if err != nil {
			if attempt == maxRetries {
				return fmt.Errorf("failed after %d attempts: %w", maxRetries+1, err)
			}
			log.Printf("Attempt %d failed for dish %s: %v", attempt+1, dish.Name, err)
			continue
		}

		// Save image record
		dishImage := &DishImage{
			DishID:   dish.ID,
			ImageURL: imageURL,
			Format:   func() *string { f := "webp"; return &f }(),
		}

		if err := s.repo.CreateDishImage(ctx, dishImage); err != nil {
			return fmt.Errorf("failed to save dish image: %w", err)
		}

		return nil
	}

	return fmt.Errorf("unreachable")
}

func (s *MenuService) createImagePrompt(dish Dish) string {
	basePrompt := fmt.Sprintf("%s, appetizing food photography, overhead shot, natural lighting, restaurant quality, no text, clean background", dish.Name)

	if dish.Description != nil && *dish.Description != "" {
		basePrompt = fmt.Sprintf("%s - %s, appetizing food photography, overhead shot, natural lighting, restaurant quality, no text, clean background",
			dish.Name, *dish.Description)
	}

	// Add style hints based on tags
	for _, tag := range dish.Tags {
		switch strings.ToLower(tag) {
		case "vegan":
			basePrompt += ", fresh vegetables"
		case "vegetarian":
			basePrompt += ", vegetarian ingredients"
		case "spicy":
			basePrompt += ", vibrant red spices"
		}
	}

	return basePrompt
}

// External API Clients
type OpenAIClient struct {
	apiKey     string
	serviceURL string
	httpClient *http.Client
}

func NewOpenAIClient(apiKey, serviceURL string) *OpenAIClient {
	return &OpenAIClient{
		apiKey:     apiKey,
		serviceURL: serviceURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *OpenAIClient) CreateChatCompletion(ctx context.Context, request OpenAIRequest) (*OpenAIResponse, error) {
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	log.Printf("OpenAI request size: %d bytes", len(jsonData))

	url := fmt.Sprintf("%s/chat/completions", c.serviceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	log.Printf("Making OpenAI API call to: %s", url)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		log.Printf("OpenAI HTTP request failed: %v", err)
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("OpenAI API response status: %d", resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("OpenAI API error response: %s", string(body))
		return nil, fmt.Errorf("OpenAI API error: %d - %s", resp.StatusCode, string(body))
	}

	log.Printf("OpenAI response body size: %d bytes", len(body))

	var response OpenAIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		log.Printf("Failed to decode OpenAI response: %v", err)
		log.Printf("Raw response: %s", string(body))
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

type ReplicateClient struct {
	apiKey     string
	serviceURL string
	httpClient *http.Client
}

func NewReplicateClient(apiKey, serviceURL string) *ReplicateClient {
	return &ReplicateClient{
		apiKey:     apiKey,
		serviceURL: serviceURL,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *ReplicateClient) GenerateImage(ctx context.Context, prompt string) (string, error) {
	// Check if we're in test mode (no valid API key)
	if c.apiKey == "r8_your-replicate-key-here" || c.apiKey == "" {
		log.Printf("Using test mode for image generation - returning placeholder image for prompt: %s", prompt)
		return "https://via.placeholder.com/512x512/FFB6C1/000000?text=Sample+Dish", nil
	}

	request := ReplicateRequest{
		Input: ReplicateInput{
			Prompt:               prompt,
			AspectRatio:          "1:1",
			NumOutputs:           1,
			NumInferenceSteps:    28,
			Guidance:             3.5,
			OutputFormat:         "webp",
			OutputQuality:        80,
			GoFast:               true,
			DisableSafetyChecker: false,
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/models/black-forest-labs/flux-dev/predictions", c.serviceURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Prefer", "wait")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Replicate API error: %d - %s", resp.StatusCode, string(body))
	}

	var response ReplicateResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// If processing, poll for completion
	if response.Status == "processing" || response.Status == "starting" {
		return c.pollForCompletion(ctx, response.URLs.Get)
	}

	if response.Status == "succeeded" && len(response.Output) > 0 {
		return response.Output[0], nil
	}

	if response.Error != nil {
		return "", fmt.Errorf("Replicate error: %s", *response.Error)
	}

	return "", fmt.Errorf("unexpected response status: %s", response.Status)
}

func (c *ReplicateClient) pollForCompletion(ctx context.Context, pollURL string) (string, error) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	timeout := time.After(60 * time.Second)

	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-timeout:
			return "", fmt.Errorf("timeout waiting for image generation")
		case <-ticker.C:
			req, err := http.NewRequestWithContext(ctx, "GET", pollURL, nil)
			if err != nil {
				return "", fmt.Errorf("failed to create poll request: %w", err)
			}

			req.Header.Set("Authorization", "Bearer "+c.apiKey)

			resp, err := c.httpClient.Do(req)
			if err != nil {
				continue // Retry on network error
			}

			if resp.StatusCode == http.StatusOK {
				var response ReplicateResponse
				if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
					resp.Body.Close()
					continue
				}
				resp.Body.Close()

				if response.Status == "succeeded" && len(response.Output) > 0 {
					return response.Output[0], nil
				}

				if response.Status == "failed" {
					if response.Error != nil {
						return "", fmt.Errorf("image generation failed: %s", *response.Error)
					}
					return "", fmt.Errorf("image generation failed")
				}
			} else {
				resp.Body.Close()
			}
		}
	}
}

// HTTP Handlers
type Handler struct {
	menuService *MenuService
	config      *Config
}

func NewHandler(menuService *MenuService, config *Config) *Handler {
	return &Handler{
		menuService: menuService,
		config:      config,
	}
}

func (h *Handler) CreateMenu(w http.ResponseWriter, r *http.Request) {
	// Parse multipart form
	if err := r.ParseMultipartForm(h.config.MaxImageSize); err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "No image file provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read file data
	imageData, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "Failed to read image", http.StatusInternalServerError)
		return
	}

	// Validate file size
	if int64(len(imageData)) > h.config.MaxImageSize {
		http.Error(w, "Image too large", http.StatusBadRequest)
		return
	}

	// Validate image type (basic check)
	if !isValidImage(imageData) {
		http.Error(w, "Invalid image format", http.StatusBadRequest)
		return
	}

	// Create menu
	menu, err := h.menuService.CreateMenu(r.Context(), imageData)
	if err != nil {
		log.Printf("Failed to create menu: %v", err)
		http.Error(w, "Failed to create menu", http.StatusInternalServerError)
		return
	}

	// Return response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(menu)
}

func (h *Handler) GetMenu(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	menuID := vars["id"]

	if menuID == "" {
		http.Error(w, "Menu ID required", http.StatusBadRequest)
		return
	}

	menu, err := h.menuService.GetMenu(r.Context(), menuID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, "Menu not found", http.StatusNotFound)
			return
		}
		log.Printf("Failed to get menu: %v", err)
		http.Error(w, "Failed to get menu", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(menu)
}

func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

// Utilities
func isValidImage(data []byte) bool {
	if len(data) < 4 {
		return false
	}

	// Check for JPEG
	if data[0] == 0xFF && data[1] == 0xD8 {
		return true
	}

	// Check for PNG
	if len(data) >= 8 &&
		data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 &&
		data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A {
		return true
	}

	return false
}

func loadConfig() *Config {
	config := &Config{
		Port:                getEnv("PORT", "8080"),
		DatabaseURL:         getDatabaseURL(),
		OpenAIAPIKey:        getOpenAIAPIKey(),
		OpenAIServiceURL:    getOpenAIServiceURL(),
		ReplicateAPIKey:     getReplicateAPIKey(),
		ReplicateServiceURL: getReplicateServiceURL(),
		MaxImageSize:        10 * 1024 * 1024, // 10MB
		ProcessingTimeout:   90 * time.Second,
	}

	if maxConcurrent := getEnv("MAX_CONCURRENT_IMAGES", "5"); maxConcurrent != "" {
		if val, err := strconv.Atoi(maxConcurrent); err == nil {
			config.MaxConcurrentImages = val
		} else {
			config.MaxConcurrentImages = 5
		}
	} else {
		config.MaxConcurrentImages = 5
	}

	return config
}

func getDatabaseURL() string {
	// Check if DATABASE_URL is directly provided (for local development)
	if databaseURL := getEnv("DATABASE_URL", ""); databaseURL != "" {
		return databaseURL
	}

	// Check for Choreo connection environment variables
	hostname := getEnv("CHOREO_CONNECTION_MENU_BACKEND_DEFAULTDB_HOSTNAME", "")
	port := getEnv("CHOREO_CONNECTION_MENU_BACKEND_DEFAULTDB_PORT", "")
	username := getEnv("CHOREO_CONNECTION_MENU_BACKEND_DEFAULTDB_USERNAME", "")
	password := getEnv("CHOREO_CONNECTION_MENU_BACKEND_DEFAULTDB_PASSWORD", "")
	database := getEnv("CHOREO_CONNECTION_MENU_BACKEND_DEFAULTDB_DATABASENAME", "")

	if hostname != "" && port != "" && username != "" && password != "" && database != "" {
		return fmt.Sprintf("postgresql://%s:%s@%s:%s/%s?sslmode=require", 
			username, password, hostname, port, database)
	}

	return ""
}

func getOpenAIAPIKey() string {
	// Check for Choreo connection environment variable first
	if key := getEnv("CHOREO_OPENAI_CONNECTION_OPENAI_API_KEY", ""); key != "" {
		return key
	}
	// Fallback to direct environment variable (for local development)
	return getEnv("OPENAI_API_KEY", "")
}

func getOpenAIServiceURL() string {
	// Check for Choreo connection environment variable first
	if url := getEnv("CHOREO_OPENAI_CONNECTION_SERVICEURL", ""); url != "" {
		return url
	}
	// Fallback to direct environment variable (for local development)
	return getEnv("OPENAI_SERVICE_URL", "https://api.openai.com/v1")
}

func getReplicateAPIKey() string {
	// Check for Choreo connection environment variable first
	if key := getEnv("CHOREO_REPLICATE_CONNECTION_APIKEY", ""); key != "" {
		return key
	}
	// Fallback to direct environment variable (for local development)
	return getEnv("REPLICATE_API_KEY", "")
}

func getReplicateServiceURL() string {
	// Check for Choreo connection environment variable first
	if url := getEnv("CHOREO_REPLICATE_CONNECTION_SERVICEURL", ""); url != "" {
		return url
	}
	// Fallback to direct environment variable (for local development)
	return getEnv("REPLICATE_SERVICE_URL", "https://api.replicate.com/v1")
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func initDatabase(databaseURL string) (*sql.DB, error) {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Run migrations
	if err := runMigrations(db); err != nil {
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	return db, nil
}

func runMigrations(db *sql.DB) error {
	migrations := []string{
		`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`,

		`CREATE TABLE IF NOT EXISTS menu (
			id UUID PRIMARY KEY,
			original_image_url TEXT NULL,
			status TEXT NOT NULL CHECK (status IN ('PENDING','PROCESSING','COMPLETE','FAILED')),
			created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
			failure_reason TEXT NULL
		);`,

		`CREATE TABLE IF NOT EXISTS dish (
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
		);`,

		`CREATE TABLE IF NOT EXISTS dish_image (
			dish_id UUID PRIMARY KEY REFERENCES dish(id) ON DELETE CASCADE,
			image_url TEXT NOT NULL,
			width INT NULL,
			height INT NULL,
			format TEXT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now()
		);`,

		`CREATE INDEX IF NOT EXISTS idx_menu_status ON menu(status);`,
		`CREATE INDEX IF NOT EXISTS idx_dish_menu_id_sort ON dish(menu_id, sort_order);`,
	}

	for _, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("failed to execute migration: %w", err)
		}
	}

	return nil
}

func main() {
	log.Println("Starting Menu Gen Backend...")

	// Load configuration
	config := loadConfig()

	// Validate required config
	if config.DatabaseURL == "" {
		log.Fatal("DATABASE_URL environment variable is required")
	}
	if config.OpenAIAPIKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}
	if config.ReplicateAPIKey == "" {
		log.Fatal("REPLICATE_API_KEY environment variable is required")
	}

	// Initialize database
	db, err := initDatabase(config.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// Initialize clients and services
	repo := NewRepository(db)
	openAIClient := NewOpenAIClient(config.OpenAIAPIKey, config.OpenAIServiceURL)
	replicateClient := NewReplicateClient(config.ReplicateAPIKey, config.ReplicateServiceURL)
	menuService := NewMenuService(repo, openAIClient, replicateClient, config)
	handler := NewHandler(menuService, config)

	// Setup routes
	r := mux.NewRouter()

	// API routes
	api := r.PathPrefix("/api").Subrouter()
	api.HandleFunc("/health", handler.HealthCheck).Methods("GET")
	api.HandleFunc("/menu", handler.CreateMenu).Methods("POST")
	api.HandleFunc("/menu/{id}", handler.GetMenu).Methods("GET")

	// Setup CORS
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})

	// Start server
	addr := ":" + config.Port
	log.Printf("Server starting on %s", addr)
	log.Printf("Max concurrent images: %d", config.MaxConcurrentImages)
	log.Printf("Processing timeout: %v", config.ProcessingTimeout)

	if err := http.ListenAndServe(addr, c.Handler(r)); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
