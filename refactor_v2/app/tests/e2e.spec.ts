import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

const ANH_MAU = path.resolve(__dirname, '../static/examples/gay_IMG0002608.jpg');
const ANH_SAI = path.resolve(__dirname, '../static/examples/gay_IMG0002355.jpg');
const THU_MUC_ANH = path.resolve(__dirname, '../../screenshots');

test.beforeAll(() => {
  fs.mkdirSync(THU_MUC_ANH, { recursive: true });
});

test('trang chủ hiển thị tiêu đề BoneVQA-Prompt và trạng thái model', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/BoneVQA-Prompt/);
  await expect(page.locator('h1')).toHaveText('BoneVQA-Prompt');
  await expect(page.getByText('Ảnh X-quang & Visual Prompt')).toBeVisible();
  await expect(page.getByText('Hội thoại với ảnh')).toBeVisible();
  await expect(page.getByText(/Chế độ mô phỏng|Model đã nạp/)).toBeVisible();
  await page.screenshot({ path: path.join(THU_MUC_ANH, '01_trang_chu.png'), fullPage: true });
});

test('API health và examples trả về hợp lệ', async ({ request }) => {
  const h = await request.get('/api/health');
  expect(h.ok()).toBeTruthy();
  const hd = await h.json();
  expect(hd.status).toBe('ok');
  expect(hd).toHaveProperty('model_loaded');
  expect(hd).toHaveProperty('device');
  const e = await request.get('/api/examples');
  expect(e.ok()).toBeTruthy();
  const ed = await e.json();
  expect(ed.examples.some((x: any) => x.name === 'gay_IMG0002608.jpg')).toBeTruthy();
});

test('upload ảnh, phân vùng SAM, hỏi đáp và nhận câu trả lời', async ({ page }) => {
  await page.goto('/');
  await page.setInputFiles('[data-testid="file-input"]', ANH_MAU);
  await expect(page.locator('[data-testid="canvas"]')).toBeVisible();
  await expect(page.getByText(/gay_IMG0002608\.jpg/)).toBeVisible();

  const canvas = page.locator('[data-testid="canvas"]');
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Không lấy được kích thước canvas');
  await page.mouse.move(box.x + box.width * 0.32, box.y + box.height * 0.34);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.50, box.y + box.height * 0.60, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByText(/Đã vẽ:\s*1/)).toBeVisible();

  await page.click('[data-testid="btn-segment"]');
  await expect(page.locator('[data-testid="img-overlay"]')).toBeVisible({ timeout: 180000 });
  await expect(page.getByText(/Vùng đã phân đoạn/)).toBeVisible({ timeout: 180000 });
  await page.screenshot({ path: path.join(THU_MUC_ANH, '02_sau_upload.png'), fullPage: true });

  await page.fill('[data-testid="question-input"]', 'Ảnh này có gãy xương không?');
  await page.click('[data-testid="btn-ask"]');
  const traLoi = page.locator('[data-role="assistant"]').first();
  await expect(traLoi).toBeVisible();
  await expect(traLoi.getByText(/^(Đóng|Mở)$/)).toBeVisible();
  await expect(traLoi.getByText(/Tin cậy/)).toBeVisible();
  await expect(traLoi.getByText(/ms$/)).toBeVisible();
  await expect(page.locator('[data-testid="img-prompt"]')).toBeVisible();
  await expect(page.locator('[data-testid="img-lens"]')).toBeVisible();

  await page.getByRole('button', { name: 'Is there a fracture?' }).click();
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(2);
  await page.screenshot({ path: path.join(THU_MUC_ANH, '03_sau_hoi.png'), fullPage: true });
});

test('câu hỏi tiếng Việt được chuyển ngữ và trả lời bằng tiếng Việt', async ({ page }) => {
  await page.goto('/');
  await page.setInputFiles('[data-testid="file-input"]', ANH_MAU);
  await expect(page.locator('[data-testid="canvas"]')).toBeVisible();

  await page.fill('[data-testid="question-input"]', 'Ảnh này có gãy xương không?');
  await page.click('[data-testid="btn-ask"]');
  const traLoi = page.locator('[data-role="assistant"]').first();
  await expect(traLoi).toBeVisible();
  await expect(traLoi.getByText('Đóng')).toBeVisible();
  await expect(traLoi.locator('p').first()).toHaveText(/^(có|không)$/);

  await page.fill('[data-testid="question-input"]', 'Đây là vùng cơ thể nào?');
  await page.click('[data-testid="btn-ask"]');
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(2);
  await expect(page.locator('[data-role="assistant"] [data-testid="answer-en"]').first()).toBeVisible();
  await page.screenshot({ path: path.join(THU_MUC_ANH, '04_hoi_tieng_viet.png'), fullPage: true });
});

test('API /api/ask trả về cả bản tiếng Anh của câu hỏi và câu trả lời', async ({ request }) => {
  const r = await request.post('/api/ask', {
    multipart: {
      image: { name: 'gay_IMG0002608.jpg', mimeType: 'image/jpeg', buffer: fs.readFileSync(ANH_MAU) },
      question: 'Có bao nhiêu vị trí gãy?',
      prompt_kind: 'contour',
    },
  });
  expect(r.ok()).toBeTruthy();
  const d = await r.json();
  expect(d.question_en).toBe('How many fracture sites are visible?');
  expect(d).toHaveProperty('answer_en');
  expect(typeof d.answer).toBe('string');
  expect(d.answer.length).toBeGreaterThan(0);
});

test('bác sĩ xác nhận kết quả và phản hồi được ghi lại', async ({ page, request }) => {
  const truoc = await (await request.get('/api/phan-hoi/thong-ke')).json();

  await page.goto('/');
  await page.setInputFiles('[data-testid="file-input"]', ANH_MAU);
  await expect(page.locator('[data-testid="canvas"]')).toBeVisible();
  await page.fill('[data-testid="question-input"]', 'Is there a fracture?');
  await page.click('[data-testid="btn-ask"]');

  const traLoi = page.locator('[data-role="assistant"]').first();
  await expect(traLoi).toBeVisible();
  await expect(traLoi.locator('[data-testid="khoi-phan-hoi"]')).toBeVisible();
  await expect(traLoi.getByText('Bác sĩ xác nhận:')).toBeVisible();

  await traLoi.locator('[data-testid="btn-dung"]').click();
  await expect(traLoi.locator('[data-testid="da-ghi-phan-hoi"]')).toBeVisible();
  await page.screenshot({ path: path.join(THU_MUC_ANH, '05_phan_hoi_xai.png'), fullPage: true });

  const sau = await (await request.get('/api/phan-hoi/thong-ke')).json();
  expect(sau.tong).toBeGreaterThan(truoc.tong);
  expect(sau.theo_danh_gia.dung).toBeGreaterThan(truoc.theo_danh_gia.dung || 0);
});

test('chọn Sai thì hiện ô nhập đáp án đúng và ghi kèm', async ({ page }) => {
  await page.goto('/');
  await page.setInputFiles('[data-testid="file-input"]', ANH_MAU);
  await page.fill('[data-testid="question-input"]', 'Which body region is shown in this X-ray?');
  await page.click('[data-testid="btn-ask"]');

  const traLoi = page.locator('[data-role="assistant"]').first();
  await expect(traLoi).toBeVisible();
  await traLoi.locator('[data-testid="btn-sai"]').click();
  await expect(traLoi.locator('[data-testid="input-dap-an-dung"]')).toBeVisible();
  await traLoi.locator('[data-testid="input-dap-an-dung"]').fill('leg');
  await traLoi.locator('[data-testid="btn-luu-dap-an"]').click();
  await expect(traLoi.locator('[data-testid="da-ghi-phan-hoi"]')).toBeVisible();
});

test('trường hợp mô hình trả lời sai với độ tin cậy thấp', async ({ page }) => {
  await page.goto('/');
  await page.setInputFiles('[data-testid="file-input"]', ANH_SAI);
  await expect(page.locator('[data-testid="canvas"]')).toBeVisible();
  await expect(page.getByText(/gay_IMG0002355\.jpg/)).toBeVisible();

  await page.getByRole('button', { name: 'Is there a fracture?' }).click();
  const traLoi = page.locator('[data-role="assistant"]').first();
  await expect(traLoi).toBeVisible();
  await expect(traLoi.getByText(/Tin cậy/)).toBeVisible();
  const tinCay = await traLoi.getByText(/Tin cậy/).innerText();
  const phanTram = parseInt(tinCay.replace(/\D/g, ''), 10);
  expect(phanTram).toBeLessThan(70);
  await page.screenshot({ path: path.join(THU_MUC_ANH, '06_truong_hop_sai.png'), fullPage: true });
});
