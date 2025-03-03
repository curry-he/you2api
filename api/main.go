package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/google/uuid"
)

// TokenCount 定义了 token 计数的结构
type TokenCount struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

const (
	MaxContextTokens = 2000 // 最大上下文 token 数
)

// YouChatResponse 定义了从 You.com API 接收的单个 token 的结构。
type YouChatResponse struct {
	YouChatToken string `json:"youChatToken"`
}

// OpenAIStreamResponse 定义了 OpenAI API 流式响应的结构。
type OpenAIStreamResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// Choice 定义了 OpenAI 流式响应中 choices 数组的单个元素的结构。
type Choice struct {
	Delta        Delta  `json:"delta"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// Delta 定义了流式响应中表示增量内容的结构。
type Delta struct {
	Content string `json:"content"`
}

// OpenAIRequest 定义了 OpenAI API 请求体的结构。
type OpenAIRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Model    string    `json:"model"`
}

// Message 定义了 OpenAI 聊天消息的结构。
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIResponse 定义了 OpenAI API 非流式响应的结构。
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
}

// OpenAIChoice 定义了 OpenAI 非流式响应中 choices 数组的单个元素的结构。
type OpenAIChoice struct {
	Message      Message `json:"message"`
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
}

// ModelResponse 定义了 /v1/models 响应的结构。
type ModelResponse struct {
	Object string        `json:"object"`
	Data   []ModelDetail `json:"data"`
}

// ModelDetail 定义了模型列表中单个模型的详细信息。
type ModelDetail struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// originalModel 存储原始请求中使用的 OpenAI 模型名称
var originalModel string

// NonceResponse 定义了获取 nonce 时的响应结构
type NonceResponse struct {
	Uuid string
}

// UploadResponse 定义了文件上传后的响应结构
type UploadResponse struct {
	Filename     string `json:"filename"`
	UserFilename string `json:"user_filename"`
}

// 定义最大查询长度（如有需要）
const MaxQueryLength = 2000

// -------------------
// 动态模型映射相关：通过 goquery 获取 __NEXT_DATA__ 中模型列表
// -------------------

// getDynamicModelMap 通过请求 https://you.com 页面，利用 goquery 解析 id 为 "__NEXT_DATA__" 的 <script> 标签内容，
// 并提取其中 "aiModels" 数组，从而构造 OpenAI 模型归一化名称到 You.com 模型 ID 的映射。
func getDynamicModelMap(dsToken string) (map[string]string, error) {
	req, err := http.NewRequest("GET", "https://you.com", nil)
	if err != nil {
		return nil, err
	}
	// 如果提供了 DS token，则在请求头中设置 Cookie
	if dsToken != "" {
		req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// 使用 goquery 从响应中构建文档对象
	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	// 选择 id 为 "__NEXT_DATA__" 的 <script> 标签，并获取其中的文本内容（即 JSON 数据）
	jsonContent := doc.Find("script#__NEXT_DATA__").Text()
	if jsonContent == "" {
		return nil, fmt.Errorf("未找到 __NEXT_DATA__ 脚本")
	}

	// 定义用于解析 JSON 的结构体
	var nextData struct {
		Props struct {
			PageProps struct {
				AiModels []struct {
					ID string `json:"id"`
				} `json:"aiModels"`
			} `json:"pageProps"`
		} `json:"props"`
	}
	if err := json.Unmarshal([]byte(jsonContent), &nextData); err != nil {
		return nil, fmt.Errorf("解析 __NEXT_DATA__ JSON 失败: %v", err)
	}

	dynamicMap := make(map[string]string)
	for _, model := range nextData.Props.PageProps.AiModels {
		var key string
		// 若模型 ID 以 "openai_" 开头，则去掉前缀
		if strings.HasPrefix(model.ID, "openai_") {
			key = strings.TrimPrefix(model.ID, "openai_")
		} else {
			key = model.ID
		}
		// 将下划线替换为连字符，并全部转换为小写
		key = strings.ToLower(strings.ReplaceAll(key, "_", "-"))
		dynamicMap[key] = model.ID
	}
	// 添加别名映射，例如将 "deepseek-v3" 也映射到 "deepseek-chat"
	if val, ok := dynamicMap["deepseek-v3"]; ok {
		dynamicMap["deepseek-chat"] = val
	}
	return dynamicMap, nil
}

// dynamicMapModelName 根据动态映射将 OpenAI 模型名称映射到对应的 You.com 模型 ID。
// 对传入的 OpenAI 模型名称进行归一化（小写，替换下划线为连字符，并去掉可能的 "openai-" 前缀），
// 在映射中查找相应的值，如果查不到则返回默认值 "deepseek_v3"。
func dynamicMapModelName(dynamicMap map[string]string, openAIModel string) string {
	normalized := strings.ToLower(strings.ReplaceAll(openAIModel, "_", "-"))
	if strings.HasPrefix(normalized, "openai-") {
		normalized = strings.TrimPrefix(normalized, "openai-")
	}
	if val, ok := dynamicMap[normalized]; ok {
		return val
	}
	return "deepseek_v3" // 默认模型
}

// dynamicReverseMapModelName 根据动态映射，将 You.com 模型 ID 还原为归一化后的 OpenAI 模型名称。
func dynamicReverseMapModelName(dynamicMap map[string]string, youModel string) string {
	for key, val := range dynamicMap {
		if val == youModel {
			return key
		}
	}
	return "deepseek-chat" // 默认返回
}

// ----------------------
// Handler 主入口函数
// ----------------------
func Handler(w http.ResponseWriter, r *http.Request) {
	// 处理 /v1/models 请求，返回可用模型列表（这里仅以动态映射中的键作为示例返回）
	if r.URL.Path == "/v1/models" || r.URL.Path == "/api/v1/models" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		// 获取 DS token（可选，根据需要验证 Authorization）
		authHeader := r.Header.Get("Authorization")
		var dsToken string
		if strings.HasPrefix(authHeader, "Bearer ") {
			dsToken = strings.TrimPrefix(authHeader, "Bearer ")
		}

		dynamicModelMap, err := getDynamicModelMap(dsToken)
		if err != nil {
			fmt.Printf("获取动态模型映射失败: %v\n", err)
			http.Error(w, "Failed to fetch dynamic model mapping", http.StatusInternalServerError)
			return
		}

		models := make([]ModelDetail, 0, len(dynamicModelMap))
		created := time.Now().Unix()
		for modelID := range dynamicModelMap {
			models = append(models, ModelDetail{
				ID:      modelID,
				Object:  "model",
				Created: created,
				OwnedBy: "you.com",
			})
		}

		response := ModelResponse{
			Object: "list",
			Data:   models,
		}

		json.NewEncoder(w).Encode(response)
		return
	}

	// 处理非 /v1/chat/completions 请求（服务状态检查）
	if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/none/v1/chat/completions" && r.URL.Path != "/such/chat/completions" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "You2Api Service Running...",
			"message": "MoLoveSze...",
		})
		return
	}

	// 设置 CORS 头部
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "*")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// 验证 Authorization 头部
	authHeader := r.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "Bearer ") {
		http.Error(w, "Missing or invalid authorization header", http.StatusUnauthorized)
		return
	}
	dsToken := strings.TrimPrefix(authHeader, "Bearer ")

	// 解析 OpenAI 请求体
	var openAIReq OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	originalModel = openAIReq.Model

	// 将所有 system 消息合并为第一条 user 消息
	openAIReq.Messages = convertSystemToUser(openAIReq.Messages)

	// 获取最新动态模型映射（动态获取页面中的 __NEXT_DATA__）
	dynamicModelMap, err := getDynamicModelMap(dsToken)
	if err != nil {
		fmt.Printf("获取动态模型映射失败: %v\n", err)
		http.Error(w, "Failed to fetch dynamic model mapping", http.StatusInternalServerError)
		return
	}

	// 构建聊天历史和文件上传逻辑
	var chatHistory []map[string]interface{}
	var sources []map[string]interface{}
	var lastAssistantMessage string

	// 处理历史消息（不包含最后一条）
	for _, msg := range openAIReq.Messages[:len(openAIReq.Messages)-1] {
		if msg.Role == "user" {
			tokens, err := countTokens([]Message{msg})
			if err != nil {
				http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
				return
			}

			if tokens > MaxContextTokens {
				nonceResp, err := getNonce(dsToken)
				if err != nil {
					fmt.Printf("获取 nonce 失败: %v\n", err)
					http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
					return
				}
				tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
				if err := os.WriteFile(tempFile, []byte(msg.Content), 0644); err != nil {
					fmt.Printf("创建临时文件失败: %v\n", err)
					http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
					return
				}
				defer os.Remove(tempFile)

				uploadResp, err := uploadFile(dsToken, tempFile)
				if err != nil {
					fmt.Printf("上传文件失败: %v\n", err)
					http.Error(w, "Failed to upload file", http.StatusInternalServerError)
					return
				}
				sources = append(sources, map[string]interface{}{
					"source_type":   "user_file",
					"filename":      uploadResp.Filename,
					"user_filename": uploadResp.UserFilename,
					"size_bytes":    len(msg.Content),
				})
				chatHistory = append(chatHistory, map[string]interface{}{
					"question": fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename),
					"answer":   "",
				})
			} else {
				chatHistory = append(chatHistory, map[string]interface{}{
					"question": msg.Content,
					"answer":   "",
				})
			}
		} else if msg.Role == "assistant" {
			lastAssistantMessage = msg.Content
		}
	}

	// 如果存在最后一条 assistant 消息，则添加到聊天历史中
	if lastAssistantMessage != "" {
		chatHistory = append(chatHistory, map[string]interface{}{
			"question": "",
			"answer":   lastAssistantMessage,
		})
	}

	chatHistoryJSON, _ := json.Marshal(chatHistory)

	// 新建调用 You.com API 的请求
	youReq, _ := http.NewRequest("GET", "https://you.com/api/streamingSearch", nil)
	chatId := uuid.New().String()
	conversationTurnId := uuid.New().String()
	traceId := fmt.Sprintf("%s|%s|%s", chatId, conversationTurnId, time.Now().Format(time.RFC3339))

	// 处理最后一条消息
	lastMessage := openAIReq.Messages[len(openAIReq.Messages)-1]
	lastMessageTokens, err := countTokens([]Message{lastMessage})
	if err != nil {
		http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
		return
	}

	q := youReq.URL.Query()
	q.Add("page", "1")
	q.Add("count", "10")
	q.Add("safeSearch", "Off")
	q.Add("mkt", "en-US")
	q.Add("enable_worklow_generation_ux", "true")
	q.Add("domain", "youchat")
	q.Add("use_personalization_extraction", "true")
	q.Add("queryTraceId", chatId)
	q.Add("chatId", chatId)
	q.Add("conversationTurnId", conversationTurnId)
	q.Add("pastChatLength", fmt.Sprintf("%d", len(chatHistory)))
	q.Add("selectedChatMode", "custom")
	// 使用动态映射将 OpenAI 模型名称转换为 You.com 模型 ID
	q.Add("selectedAiModel", dynamicMapModelName(dynamicModelMap, openAIReq.Model))
	q.Add("enable_agent_clarification_questions", "true")
	q.Add("traceId", traceId)
	q.Add("use_nested_youchat_updates", "true")

	if lastMessageTokens > MaxContextTokens {
		nonceResp, err := getNonce(dsToken)
		if err != nil {
			fmt.Printf("获取 nonce 失败: %v\n", err)
			http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
			return
		}
		tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
		if err := os.WriteFile(tempFile, []byte(lastMessage.Content), 0644); err != nil {
			fmt.Printf("创建临时文件失败: %v\n", err)
			http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
			return
		}
		defer os.Remove(tempFile)

		uploadResp, err := uploadFile(dsToken, tempFile)
		if err != nil {
			fmt.Printf("上传文件失败: %v\n", err)
			http.Error(w, "Failed to upload file", http.StatusInternalServerError)
			return
		}
		sources = append(sources, map[string]interface{}{
			"source_type":   "user_file",
			"filename":      uploadResp.Filename,
			"user_filename": uploadResp.UserFilename,
			"size_bytes":    len(lastMessage.Content),
		})
		sourcesJSON, _ := json.Marshal(sources)
		q.Add("sources", string(sourcesJSON))
		q.Add("q", fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename))
	} else {
		if len(sources) > 0 {
			sourcesJSON, _ := json.Marshal(sources)
			q.Add("sources", string(sourcesJSON))
		}
		q.Add("q", lastMessage.Content)
	}

	q.Add("chat", string(chatHistoryJSON))
	youReq.URL.RawQuery = q.Encode()

	// 打印请求信息（日志）
	fmt.Printf("\n=== 完整请求信息 ===\n")
	fmt.Printf("请求 URL: %s\n", youReq.URL.String())
	fmt.Printf("请求头:\n")
	for key, values := range youReq.Header {
		fmt.Printf("%s: %v\n", key, values)
	}

	// 设置请求所需头部信息
	youReq.Header = http.Header{
		"sec-ch-ua-platform":         {"Windows"},
		"Cache-Control":              {"no-cache"},
		"sec-ch-ua":                  {`"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"`},
		"sec-ch-ua-bitness":          {"64"},
		"sec-ch-ua-model":            {""},
		"sec-ch-ua-mobile":           {"?0"},
		"sec-ch-ua-arch":             {"x86"},
		"sec-ch-ua-full-version":     {"133.0.3065.39"},
		"Accept":                     {"text/event-stream"},
		"User-Agent":                 {"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"},
		"sec-ch-ua-platform-version": {"19.0.0"},
		"Sec-Fetch-Site":             {"same-origin"},
		"Sec-Fetch-Mode":             {"cors"},
		"Sec-Fetch-Dest":             {"empty"},
		"Host":                       {"you.com"},
	}
	cookies := getCookies(dsToken)
	var cookieStrings []string
	for name, value := range cookies {
		cookieStrings = append(cookieStrings, fmt.Sprintf("%s=%s", name, value))
	}
	youReq.Header.Add("Cookie", strings.Join(cookieStrings, ";"))
	fmt.Printf("Cookie: %s\n", strings.Join(cookieStrings, ";"))
	fmt.Printf("===================\n\n")

	client := &http.Client{}
	resp, err := client.Do(youReq)
	if err != nil {
		fmt.Printf("发送请求失败: %v\n", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	fmt.Printf("响应状态码: %d\n", resp.StatusCode)
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("错误响应内容: %s\n", string(body))
		http.Error(w, fmt.Sprintf("API returned status %d", resp.StatusCode), resp.StatusCode)
		return
	}

	// 根据 stream 参数判断使用流式或非流式处理函数
	if !openAIReq.Stream {
		handleNonStreamingResponse(w, youReq, dynamicModelMap)
		return
	}
	handleStreamingResponse(w, youReq, dynamicModelMap)
}

// getCookies 根据提供的 DS token 生成所需的 Cookie
func getCookies(dsToken string) map[string]string {
	return map[string]string{
		"guest_has_seen_legal_disclaimer": "true",
		"youchat_personalization":         "true",
		"DS":                              dsToken,
		"you_subscription":                "youpro_standard_year",
		"youpro_subscription":             "true",
		"ai_model":                        "deepseek_r1", // 示例 AI 模型
		"youchat_smart_learn":             "true",
	}
}

// handleNonStreamingResponse 处理非流式响应，并使用动态映射还原模型名称
func handleNonStreamingResponse(w http.ResponseWriter, youReq *http.Request, dynamicModelMap map[string]string) {
	client := &http.Client{
		Timeout: 60 * time.Second,
	}
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan() // 读取下一行 (data 行)
			data := scanner.Text()
			if !strings.HasPrefix(data, "data: ") {
				continue
			}
			var token YouChatResponse
			if err := json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token); err != nil {
				continue
			}
			fullResponse.WriteString(token.YouChatToken)
		}
	}

	if scanner.Err() != nil {
		http.Error(w, "Error reading response", http.StatusInternalServerError)
		return
	}

	openAIResp := OpenAIResponse{
		ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		// 利用动态映射还原 OpenAI 模型名称
		Model: dynamicReverseMapModelName(dynamicModelMap, dynamicMapModelName(dynamicModelMap, originalModel)),
		Choices: []OpenAIChoice{
			{
				Message: Message{
					Role:    "assistant",
					Content: fullResponse.String(),
				},
				Index:        0,
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(openAIResp); err != nil {
		http.Error(w, "Error encoding response", http.StatusInternalServerError)
		return
	}
}

// handleStreamingResponse 处理流式响应，并使用动态映射还原模型名称
func handleStreamingResponse(w http.ResponseWriter, youReq *http.Request, dynamicModelMap map[string]string) {
	client := &http.Client{}
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan()
			data := scanner.Text()
			var token YouChatResponse
			json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token)
			openAIResp := OpenAIStreamResponse{
				ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   dynamicReverseMapModelName(dynamicModelMap, dynamicMapModelName(dynamicModelMap, originalModel)),
				Choices: []Choice{
					{
						Delta: Delta{
							Content: token.YouChatToken,
						},
						Index:        0,
						FinishReason: "",
					},
				},
			}
			respBytes, _ := json.Marshal(openAIResp)
			fmt.Fprintf(w, "data: %s\n\n", string(respBytes))
			w.(http.Flusher).Flush()
		}
	}
}

// 获取 nonce 的函数保持不变
func getNonce(dsToken string) (*NonceResponse, error) {
	req, _ := http.NewRequest("GET", "https://you.com/api/get_nonce", nil)
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %v", err)
	}

	return &NonceResponse{
		Uuid: strings.TrimSpace(string(body)),
	}, nil
}

// 上传文件的函数保持不变
func uploadFile(dsToken, filePath string) (*UploadResponse, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	writer.Close()

	req, _ := http.NewRequest("POST", "https://you.com/api/upload", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var uploadResp UploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
		return nil, err
	}
	return &uploadResp, nil
}

// 计算消息的 token 数（简单字符估算法）
func countTokens(messages []Message) (int, error) {
	totalTokens := 0
	for _, msg := range messages {
		content := msg.Content
		englishCount := 0
		chineseCount := 0
		for _, r := range content {
			if r <= 127 {
				englishCount++
			} else {
				chineseCount++
			}
		}
		tokens := int(float64(englishCount)*0.3 + float64(chineseCount)*1)
		totalTokens += tokens + 2
	}
	return totalTokens, nil
}

// 将所有 system 消息合并为第一条 user 消息
func convertSystemToUser(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}

	var systemContent strings.Builder
	var newMessages []Message
	var systemFound bool

	for _, msg := range messages {
		if msg.Role == "system" {
			if systemContent.Len() > 0 {
				systemContent.WriteString("\n")
			}
			systemContent.WriteString(msg.Content)
			systemFound = true
		} else {
			newMessages = append(newMessages, msg)
		}
	}

	if systemFound {
		newMessages = append([]Message{{
			Role:    "user",
			Content: systemContent.String(),
		}}, newMessages...)
	}

	return newMessages
}
