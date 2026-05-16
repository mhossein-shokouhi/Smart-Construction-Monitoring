/**
 * Pi IP Registry — Google Apps Script
 *
 * Deploy as a Web App (Execute as: Me, Who has access: Anyone).
 * Create a spreadsheet first, then Extensions → Apps Script → paste this file.
 *
 * Set Script property REGISTRY_SECRET (Project settings → Script properties)
 * or edit DEFAULT_SECRET below before deploying.
 */

const SHEET_NAME = 'Pi Registry';
const DEFAULT_SECRET = 'CHANGE_ME_BEFORE_DEPLOY';

function getSecret_() {
  const fromProps = PropertiesService.getScriptProperties().getProperty('REGISTRY_SECRET');
  return (fromProps || DEFAULT_SECRET).trim();
}

function getOrCreateSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
    sheet.appendRow([
      'pi_id',
      'hostname',
      'ip',
      'interface',
      'last_seen_utc',
      'status',
    ]);
    sheet.getRange(1, 1, 1, 6).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function jsonResponse_(obj, statusCode) {
  const output = ContentService.createTextOutput(JSON.stringify(obj));
  output.setMimeType(ContentService.MimeType.JSON);
  // Apps Script Web Apps don't expose HTTP status codes directly;
  // clients should check the "ok" field in the JSON body.
  return output;
}

function validateSecret_(provided) {
  const expected = getSecret_();
  if (!provided || provided !== expected) {
    return false;
  }
  return true;
}

function findRowForPiId_(sheet, piId) {
  const data = sheet.getDataRange().getValues();
  for (let i = 1; i < data.length; i++) {
    if (String(data[i][0]) === String(piId)) {
      return i + 1; // 1-based row index
    }
  }
  return -1;
}

function doPost(e) {
  try {
    const body = e.postData && e.postData.contents
      ? JSON.parse(e.postData.contents)
      : {};

    if (!validateSecret_(body.secret)) {
      return jsonResponse_({ ok: false, error: 'unauthorized' });
    }

    const piId = body.pi_id;
    const ip = (body.ip || '').trim();
    const iface = (body.interface || '').trim();
    const hostname = (body.hostname || '').trim();

    if (piId === undefined || piId === null || piId === '') {
      return jsonResponse_({ ok: false, error: 'pi_id required' });
    }
    if (!ip) {
      return jsonResponse_({ ok: false, error: 'ip required' });
    }

    const sheet = getOrCreateSheet_();
    const now = new Date().toISOString();
    const row = findRowForPiId_(sheet, piId);

    const values = [String(piId), hostname, ip, iface, now, 'online'];

    if (row > 0) {
      // getRange(row, col, numRows, numColumns) — third arg is row COUNT, not end row
      sheet.getRange(row, 1, 1, values.length).setValues([values]);
    } else {
      sheet.appendRow(values);
    }

    return jsonResponse_({ ok: true, pi_id: String(piId), ip: ip, last_seen_utc: now });
  } catch (err) {
    return jsonResponse_({ ok: false, error: String(err) });
  }
}

function doGet(e) {
  try {
    const params = (e && e.parameter) || {};
    if (!validateSecret_(params.secret)) {
      return jsonResponse_({ ok: false, error: 'unauthorized' });
    }

    const sheet = getOrCreateSheet_();
    const data = sheet.getDataRange().getValues();
    const headers = data[0] || [];
    const rows = [];

    for (let i = 1; i < data.length; i++) {
      const obj = {};
      for (let j = 0; j < headers.length; j++) {
        obj[headers[j]] = data[i][j];
      }
      rows.push(obj);
    }

    return jsonResponse_({ ok: true, pis: rows });
  } catch (err) {
    return jsonResponse_({ ok: false, error: String(err) });
  }
}
